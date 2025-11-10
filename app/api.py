# app/api.py
import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pipeline import RecommenderService
from app.jd_fetch import fetch_text_from_url

DATASET_PATH = os.getenv("DATASET_PATH", "data/assessments_catalog.json")

app = FastAPI(title="SHL Assessment Recommender")

service = RecommenderService(DATASET_PATH)

class RecommendIn(BaseModel):
    query: str | None = None
    jd_url: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(inp: RecommendIn):
    if not inp.query and not inp.jd_url:
        raise HTTPException(status_code=400, detail="Provide either 'query' or 'jd_url'.")

    try:
        text = inp.query or fetch_text_from_url(inp.jd_url)
        results = service.recommend_v2(text)
        # respond with only 10 items, each has at least name, url
        # (We already include extra fields too, which is fine; the spec mandates at least these.)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")
