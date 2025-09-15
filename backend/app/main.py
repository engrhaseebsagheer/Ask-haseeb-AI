from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router
from .utils.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG API (OpenAI + Pinecone) for Ask Haseeb AI",
    version="1.0.0"
)

# CORS
origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes under /api
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running", "docs": "/docs"}
