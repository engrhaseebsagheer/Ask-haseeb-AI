from fastapi import APIRouter, HTTPException
from ..models.schema import QueryRequest, AskResponse, MatchChunk, HealthResponse
from ..services.rag_service import rag_answer
from ..utils.config import get_settings


from fastapi import APIRouter
from ..models.schema import HealthResponse
from ..utils.config import get_settings
from ..services.auto_ingest import process_new_drive_files

router = APIRouter()
settings = get_settings()

@router.post("/ingest/run")
def run_ingest_now():
    result = process_new_drive_files()
    return {"status": "ok", "result": result}

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "app": settings.APP_NAME}

@router.post("/ask", response_model=AskResponse)
def ask(req: QueryRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    answer, matches = rag_answer(req.text)

    retrieved = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        retrieved.append(MatchChunk(
            score=float(m.get("score", 0)),
            title=md.get("title"),
            source=md.get("source"),
            text=md.get("text")
        ))

    return AskResponse(query=req.text, answer=answer, retrieved=retrieved)
