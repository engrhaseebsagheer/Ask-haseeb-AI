from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# ------------------------------
# CORS Settings
# ------------------------------
origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# API Routes
# ------------------------------
app.include_router(api_router, prefix="/api")

# ------------------------------
# Serve Frontend (Static Files)
# ------------------------------
# Serve React/Vue/HTML from root but after API routes
app.mount("/", StaticFiles(directory="frontend/src/pages", html=True), name="frontend")

# ------------------------------
# Scheduler for Auto-Ingest
# ------------------------------
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def _startup():
    try:
        result = process_new_drive_files()
        print(f"[ingest@startup] {result}")
    except FileNotFoundError:
        print("[ingest@startup][warn] Google service account JSON not found. Skipping ingest.")
    except Exception as e:
        print(f"[ingest@startup][error] {e}")

    poll_interval = getattr(settings, "INGEST_POLL_INTERVAL_MINUTES", 10)
    scheduler.add_job(
        process_new_drive_files,
        "interval",
        minutes=poll_interval,
        id="drive_ingest_job",
        replace_existing=True
    )
    scheduler.start()

@app.on_event("shutdown")
async def _shutdown():
    scheduler.shutdown(wait=False)

# ------------------------------
# Health Check Endpoint
# ------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "docs": "/docs"}
