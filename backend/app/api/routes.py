# backend/app/api/routes.py
import json
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.schema import QueryRequest
from ..services.rag_service import rag_answer, rag_stream
from ..utils.config import get_settings

router = APIRouter()
settings = get_settings()

# ------------------------------
# Rate limit: every question costs money at OpenAI, so cap each visitor.
# In-memory, per client IP. Resets when the app restarts.
# ------------------------------
PER_MINUTE = 10
PER_DAY = 40
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Nginx passes the real visitor address in these headers.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > 86400:
        hits.popleft()
    last_minute = sum(1 for t in hits if now - t <= 60)
    if last_minute >= PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many questions in a minute. Wait a moment and try again.")
    if len(hits) >= PER_DAY:
        raise HTTPException(status_code=429, detail="Daily question limit reached. Come back tomorrow, or email Haseeb directly.")
    hits.append(now)


def _history(req: QueryRequest):
    return [{"role": t.role, "text": t.text} for t in req.history]


@router.post("/ask")
def ask(req: QueryRequest, request: Request):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    _check_rate_limit(_client_ip(request))

    try:
        result = rag_answer(req.text.strip(), req.style, _history(req))
    except Exception as e:  # OpenAI down, key invalid, quota exhausted, Pinecone error
        print(f"[ask][error] {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now. Please try again later.")

    return {"answer": result["answer"], "sources": result["sources"]}


@router.post("/ask/stream")
def ask_stream(req: QueryRequest, request: Request):
    """Same as /ask, but the answer arrives piece by piece as server-sent events."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    _check_rate_limit(_client_ip(request))
    query, style, history = req.text.strip(), req.style, _history(req)

    def events():
        try:
            for kind, payload in rag_stream(query, style, history):
                key = "sources" if kind == "sources" else "text"
                yield "data: " + json.dumps({"type": kind, key: payload}, ensure_ascii=False) + "\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as e:
            print(f"[ask/stream][error] {type(e).__name__}: {e}")
            yield "data: " + json.dumps({"type": "error", "text": "The assistant is unavailable right now. Please try again later."}) + "\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # tells Nginx not to hold the stream back
    )
