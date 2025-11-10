# summarizer_gemini.py
import os, re, json, time
from typing import List, Dict, Optional
import google.generativeai as genai

# ---------------- CONFIG ----------------

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")

SYSTEM_INSTRUCTION = """
You will receive a hiring query or job description in multiple parts.
Ingest each PART silently. Do NOT produce the final answer until you receive 'END'.
When you receive 'END', output ONLY valid JSON with exactly these keys (all values as strings):

{
  "looking_for": "string",     // 1–3 sentence summary mentioning skills to test and target job roles/department/position
  "instructions": "string",    // Concrete instructions the user implied: time limit(s), sittings, max questions, test type(s), scoring/rules, required language(s), job-level constraints, etc.
  "job_level": "string",       // Normalize to one of: entry-level, mid, senior, executive, director, mid-professional, unknown
  "language": "string"         // Explicit assessment language(s) mentioned; empty if not stated.
}

Heuristics:
- If skills are not directly present, infer the most relevant skills for the job role/position; mention them in 'looking_for'.
- Do NOT mention job level or candidate experience inside 'looking_for'; focus on skills / job role / what to assess.
- Ensure key terms explicitly present in the query/JD appear in 'looking_for'. Expand unclear abbreviations in 'looking_for'.
- In 'instructions', capture constraints like time limits, test types, required language(s), job level constraints, etc., as short sentences (comma-separated if multiple).
- If a language is stated, also reflect it in 'language'. If not stated, 'language' must be an empty string (do NOT guess).
- Normalize job_level strictly to: entry-level, mid, senior, executive, director, mid-professional, unknown.
- Output must be compact; no newlines inside values; no explanations outside JSON.
Until 'END', reply only with a tiny ACK JSON like {"ack":"part i/N received"}.
"""

# ---------------- INIT ----------------
def _init_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

# ---------------- SAFE HELPERS ----------------
def _extract_text(resp) -> Optional[str]:
    """Safely extract the first text part from a Gemini response."""
    try:
        for cand in (getattr(resp, "candidates", []) or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", []) or []):
                if getattr(part, "text", None):
                    return part.text
        try:
            return resp.text
        except Exception:
            return None
    except Exception:
        return None

def _force_json(raw: str) -> Dict[str, str]:
    m = re.search(r"\{[\s\S]*?\}", raw)
    if not m:
        raise ValueError("No JSON object found in response")
    data = json.loads(m.group(0))
    out = {
        "looking_for": str(data.get("looking_for", "")).strip(),
        "instructions": str(data.get("instructions", "")).strip(),
        "job_level": str(data.get("job_level", "")).strip(),
        "language": str(data.get("language", "")).strip(),
    }

    # --- Normalize job_level to: entry-level, mid, senior, executive, director, mid-professional, unknown
    jl = out["job_level"].lower()
    if any(k in jl for k in ["entry", "fresher", "junior", "graduate"]):
        out["job_level"] = "entry-level"
    elif "mid-professional" in jl:
        out["job_level"] = "mid-professional"
    elif any(k in jl for k in ["mid", "intermediate"]):
        out["job_level"] = "mid"
    elif "director" in jl:
        out["job_level"] = "director"
    elif any(k in jl for k in ["executive", "cxo", "chief", "vp", "svp", "evp"]):
        out["job_level"] = "executive"
    elif any(k in jl for k in ["senior", "lead", "principal", "staff"]):
        out["job_level"] = "senior"
    else:
        out["job_level"] = "unknown"

    # language: keep only if explicitly provided; else empty string
    if not out["language"] or out["language"].lower() in {"n/a", "na", "none", "unknown"}:
        out["language"] = ""

    # compact values (no newlines)
    for k in list(out.keys()):
        out[k] = re.sub(r"\s*\n+\s*", " ", out[k]).strip()

    return out

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf: chunks.append(buf)
            buf = p
    if buf: chunks.append(buf)
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final

def _with_retry_send(chat, msg, cfg, tries=3, backoff=2.0):
    last_err = None
    for t in range(tries):
        try:
            return chat.send_message(msg, generation_config=cfg)
        except Exception as e:
            last_err = e
            if t < tries - 1:
                time.sleep(backoff * (2 ** t))
            else:
                raise last_err

# ---------------- MAIN (ITERATIVE) ----------------
def simplify_query_with_gemini_iterative(text: str, verbose: bool = False) -> Dict[str, str]:
    model = _init_client()
    chat = model.start_chat(history=[])

    chunks = chunk_text(text, max_chars=4000)
    N = len(chunks)

    ack_cfg = {
        "temperature": 0.0,
        "max_output_tokens": 64,
        "response_mime_type": "application/json",
    }

    for i, part in enumerate(chunks, 1):
        msg = f"PART {i}/{N}\n\n{part}\n\nACK only; final JSON after 'END'."
        resp = _with_retry_send(chat, msg, ack_cfg, tries=3, backoff=2.0)
        if verbose:
            ack = _extract_text(resp) or '{"ack":"received"}'
            print(f"[ACK] {i}/{N}: {ack}")

    final_prompt = (
        "END\n\nNow output ONLY the JSON object described earlier. "
        "No commentary. If your previous attempt was cut off, continue here and return the full JSON."
    )
    final_cfg = {
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
    }

    final = _with_retry_send(chat, final_prompt, final_cfg, tries=3, backoff=2.0)
    raw = _extract_text(final)

    if not raw:
        retry = _with_retry_send(
            chat,
            "Return the JSON now. Remember: only the JSON object with the specified keys.",
            final_cfg, tries=2, backoff=3.0
        )
        raw = _extract_text(retry)

    if not raw:
        raise RuntimeError("Model returned no text in the final step. Try reducing input size or increasing max_output_tokens.")

    return _force_json(raw)

# ---------------- PUBLIC API ----------------
def get_assessment_summary(query: str, verbose: bool = False) -> Dict[str, str]:
    """
    Takes a job description or query string, returns the simplified JSON summary:
    {
      "looking_for": "...",
      "instructions": "...",
      "job_level": "entry-level|mid|senior|executive|director|mid-professional|unknown",
      "language": "..."
    }
    """
    return simplify_query_with_gemini_iterative(query, verbose=verbose)

# ---------------- DEMO ----------------
if __name__ == "__main__":
    demo_query = "Need a Java developer who is good in collaborating with external teams and stakeholders"
    result = get_assessment_summary(demo_query, verbose=True)
    print("\nFinal Output:")
    print(json.dumps(result, indent=2))
