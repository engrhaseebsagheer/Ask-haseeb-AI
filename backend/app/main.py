from fastapi import FastAPI
from backend.app.utils.config import settings

app = FastAPI(title="Ask Haseeb AI - Day 1")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sanity")
def sanity():
    # don't leak secrets—just confirm presence
    return {
        "openai_key_set": bool(settings.openai_api_key),
        "pinecone_key_set": bool(settings.pinecone_api_key),
        "embed_model": settings.openai_embed_model,
    }
