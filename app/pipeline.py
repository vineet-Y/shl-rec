# app/pipeline.py
import json, re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import google.generativeai as genai
import os

# -------- Config (weights & constants) --------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_FOR_LLM = 25
FINAL_K       = 10
LITE_DESC_CHARS = 600

# Dataset columns (your new schema)
COL_NAME   = "name"
COL_URL    = "url"
COL_PDF    = "pdf_text"
COL_LEVEL  = "job_level"
COL_LANG   = "test_language"
COL_ADAPT  = "adaptive_support"
COL_DESC   = "description"
COL_DUR    = "duration"
COL_TYPE   = "test_type"
COL_REMOTE = "remote_support"

# Retrieval weights
W_EMB_LOOK   = 0.75
W_EMB_LEVEL  = 0.10
W_EMB_LANG   = 0.10
W_TFIDF_LOOK = 0.15

# -------------  -------------
# Import your summarizer from your module:
from app.summarizer import get_assessment_summary


#def get_assessment_summary(query_text: str) -> str:
#   raise NotImplementedError("Plug your get_assessment_summary implementation here.")

# ------------- LLM call (implement this) -------------
# --- in app/pipeline.py ---


def call_llm_reranker(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Return empty so we fall back to retrieval-only top 10
        return ""
    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 512})
        return getattr(resp, "text", "") or ""
    except Exception:
        return ""  # safe fallback → backfill ensures 10 results
  # fallback: empty -> backfill

def norm_text(x) -> str:
    if isinstance(x, list):
        x = " ".join(map(str, x))
    elif isinstance(x, dict):
        x = json.dumps(x, ensure_ascii=False)
    elif x is None:
        x = ""
    else:
        x = str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

def load_dataset_json(path: str) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data, list), "Dataset must be JSON array."
    df = pd.DataFrame(data).fillna("")
    for c in [COL_PDF, COL_LEVEL, COL_LANG, COL_NAME, COL_URL, COL_ADAPT, COL_DESC, COL_REMOTE]:
        if c in df.columns: df[c] = df[c].map(norm_text)
        else: df[c] = ""
    if COL_DUR in df.columns:
        def _to_int(v):
            try: return int(float(str(v).strip()))
            except: return v
        df[COL_DUR] = df[COL_DUR].map(_to_int)
    else:
        df[COL_DUR] = ""
    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].map(lambda x: x if isinstance(x, list)
                                        else [t.strip() for t in str(x).split(",") if t.strip()])
    else:
        df[COL_TYPE] = [[]]
    return df

# ------- Build index (embeddings + tfidf) --------
class Retriever:
    def __init__(self, dataset_path: str):
        self.df = load_dataset_json(dataset_path)
        self.model = SentenceTransformer(MODEL_NAME)

        self.emb_pdf   = self.model.encode(self.df[COL_PDF].tolist(),   convert_to_tensor=False, normalize_embeddings=True).astype("float32")
        self.emb_level = self.model.encode(self.df[COL_LEVEL].tolist(), convert_to_tensor=False, normalize_embeddings=True).astype("float32")
        self.emb_lang  = self.model.encode(self.df[COL_LANG].tolist(),  convert_to_tensor=False, normalize_embeddings=True).astype("float32")

        self.tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100_000)
        self.tfidf_mat = self.tfidf.fit_transform(self.df[COL_PDF].tolist())
        self.tfidf_mat = normalize(self.tfidf_mat, norm="l2", copy=False)

    @staticmethod
    def _cos01(x: np.ndarray) -> np.ndarray:
        return (x + 1.0) / 2.0

    def _tfidf_sim(self, qtext: str) -> np.ndarray:
        if not qtext:
            return np.zeros(self.df.shape[0], dtype="float32")
        vec = TfidfVectorizer(vocabulary=self.tfidf.vocabulary_, ngram_range=(1,2))
        qv  = vec.fit_transform([qtext])
        sims = (self.tfidf_mat @ qv.T).toarray().ravel().astype("float32")
        if sims.max() > 0: sims /= sims.max()
        return sims

    def score_candidates(self, q: Dict[str, str]) -> pd.DataFrame:
        q_look = norm_text(q.get("looking_for",""))
        q_lvl  = norm_text(q.get("job_level",""))
        q_lang = norm_text(q.get("language",""))

        e_look = self.model.encode(q_look, convert_to_tensor=False, normalize_embeddings=True) if q_look else None
        e_lvl  = self.model.encode(q_lvl,  convert_to_tensor=False, normalize_embeddings=True) if q_lvl  else None
        e_lang = self.model.encode(q_lang, convert_to_tensor=False, normalize_embeddings=True) if q_lang else None

        n = len(self.df)
        s_look = np.full(n, 0.5, dtype="float32")
        s_lvl  = np.full(n, 0.5, dtype="float32")
        s_lang = np.full(n, 0.5, dtype="float32")

        if e_look is not None: s_look = self._cos01(self.emb_pdf   @ e_look)
        if e_lvl  is not None: s_lvl  = self._cos01(self.emb_level @ e_lvl)
        if e_lang is not None: s_lang = self._cos01(self.emb_lang  @ e_lang)

        s_look_lex = self._tfidf_sim(q_look)

        final = (0.75 * s_look) + (0.10 * s_lvl) + (0.10 * s_lang) + (0.15 * s_look_lex)

        out = self.df[[COL_NAME, COL_URL, COL_PDF, COL_LEVEL, COL_LANG, COL_DESC, COL_ADAPT, COL_REMOTE, COL_DUR, COL_TYPE]].copy()
        out["score_topic"] = s_look
        out["score_level"] = s_lvl
        out["score_lang"]  = s_lang
        out["score_lex"]   = s_look_lex
        out["final_score"] = final
        out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
        return out

def _robust_parse_summary(json_text: Any) -> Dict[str, str]:
    if isinstance(json_text, dict): d = json_text
    else:
        s = str(json_text); i, j = s.find("{"), s.rfind("}")
        d = json.loads(s[i:j+1]) if (i!=-1 and j!=-1 and j>i) else {}
    for k in ["looking_for","instructions","job_level","language"]:
        d.setdefault(k, ""); 
        if not isinstance(d[k], str): d[k] = str(d[k])
    return d

def _candidate_payload_lite(row: pd.Series, idx: int) -> dict:
    return {
        "idx":          int(idx),
        "name":         row.get(COL_NAME, ""),
        "duration":     row.get(COL_DUR, ""),
        "job_level":    row.get(COL_LEVEL, ""),
        "language":     row.get(COL_LANG, ""),
        "description":  str(row.get(COL_DESC, ""))[:LITE_DESC_CHARS],
    }

def llm_select_top10_idx(updated_query: str, items: List[dict]) -> List[int]:
    instruction = f"""
You are an assessment selection assistant.
UPDATED QUERY (user intent):
{updated_query}

You will receive up to 25 candidate assessments, each with:
idx, name, duration, job_level, language, short description.

Task:
- Select EXACTLY 10 indices (idx) that best satisfy the UPDATED QUERY.
- Balance topic/skills (based on description/name), job level, and language.
- Prefer diversity when near-duplicates exist.
- If at least 10 candidates exist, you MUST return 10.

Return ONLY:
{{"selected_idx": [idx1, idx2, ...]}}
"""
    payload = {"candidates": items}
    prompt  = instruction + "\nCANDIDATES_JSON:\n" + json.dumps(payload, ensure_ascii=False)
    raw = call_llm_reranker(prompt)

    try:
        i, j = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[i:j+1]) if (i!=-1 and j!=-1 and j>i) else {}
        arr = data.get("selected_idx", [])
        out, seen = [], set()
        for v in arr:
            try:
                iv = int(str(v).strip())
                if iv not in seen:
                    out.append(iv); seen.add(iv)
            except: 
                pass
        return out[:FINAL_K]
    except:
        return []

def _yn(val: str) -> str:
    v = (val or "").strip().lower()
    if v in {"yes","true","y","1"}: return "Yes"
    if v in {"no","false","n","0"}: return "No"
    return "Yes" if "yes" in v else ("No" if "no" in v else (val or ""))

def _format_rows(df_sel: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for _, r in df_sel.iterrows():
        rows.append({
            "url":              r.get(COL_URL, ""),
            "name":             r.get(COL_NAME, ""),
            "adaptive support": _yn(r.get(COL_ADAPT, "")),
            "description":      r.get(COL_DESC, ""),
            "duration":         str(r.get(COL_DUR, "")),
            "remote_support":   _yn(r.get(COL_REMOTE, "")),
            "test_type":        r.get(COL_TYPE, []) if isinstance(r.get(COL_TYPE, []), list)
                                else [t.strip() for t in str(r.get(COL_TYPE, "")).split(",") if t.strip()]
        })
    return rows

# -------- main callable from API --------
class RecommenderService:
    def __init__(self, dataset_path: str):
        self.index = Retriever(dataset_path)

    def recommend_v2(self, raw_text: str) -> List[Dict[str, Any]]:
        summary = get_assessment_summary(raw_text)
        q = _robust_parse_summary(summary)
        ranked = self.index.score_candidates(q)
        top25  = ranked.head(TOP_K_FOR_LLM).copy().reset_index(drop=True)
        top25["__idx"] = top25.index
        updated_query = (q.get("looking_for","") + " " + q.get("instructions","")).strip()
        items_for_llm = [_candidate_payload_lite(row, int(row["__idx"])) for _, row in top25.iterrows()]

        selected_idx = llm_select_top10_idx(updated_query, items_for_llm)
        if len(selected_idx) < FINAL_K:
            seen = set(selected_idx)
            for i in top25["__idx"].tolist():
                if i not in seen:
                    selected_idx.append(int(i)); seen.add(int(i))
                if len(selected_idx) >= FINAL_K: break

        pos = {i:p for p,i in enumerate(selected_idx)}
        chosen = top25[top25["__idx"].isin(selected_idx)].copy()
        chosen["__order"] = chosen["__idx"].map(pos)
        chosen = chosen.sort_values(["__order","final_score"], ascending=[True, False]).drop(columns=["__order","__idx"])
        return _format_rows(chosen)
