# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException
from ..models.schema import QueryRequest
from ..services.rag_service import rag_answer
from ..utils.config import get_settings

router = APIRouter()
settings = get_settings()

@router.post("/ask")
def ask(req: QueryRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    result = rag_answer(req.text)

    # Extract top 3 sources for the UI
    sources = []
    for m in result["matches"][:3]:
        md = m.get("metadata", {}) or {}
        sources.append({
            "title": md.get("title") or "Untitled",
            "url": md.get("source") or "#"
        })

    return {
        "answer": result["answer"],
        "sources": sources
    }
