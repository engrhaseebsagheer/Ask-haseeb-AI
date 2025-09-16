from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .api.routes import router as api_router
from .utils.config import get_settings
from .services.auto_ingest import process_new_drive_files

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

app.include_router(api_router, prefix="/api")

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def _startup():
    # Run once at startup
    try:
        result = process_new_drive_files()
        print(f"[ingest@startup] {result}")
    except Exception as e:
        print(f"[ingest@startup][error] {e}")

    # Then schedule periodic checks
    scheduler.add_job(
        process_new_drive_files,
        "interval",
        minutes=settings.INGEST_POLL_INTERVAL_MINUTES,
        id="drive_ingest_job",
        replace_existing=True
    )
    scheduler.start()

@app.on_event("shutdown")
async def _shutdown():
    scheduler.shutdown(wait=False)

@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running", "docs": "/docs"}
