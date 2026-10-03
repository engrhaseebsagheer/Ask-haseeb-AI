import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Ask Haseeb AI")
    ENV: str = os.getenv("ENV", "dev")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # gpt-6-luna is OpenAI's most efficient current model; it only has to rewrite retrieved passages.
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")
    # It is a reasoning model. "none" keeps answers fast and cheap; raise to "low" if answers get sloppy.
    OPENAI_REASONING_EFFORT: str = os.getenv("OPENAI_REASONING_EFFORT", "none")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "")

    # Retrieval settings
    TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0"))
    INGEST_POLL_INTERVAL_MINUTES: int = 5  # default 5 min interval

    # CORS
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

@lru_cache
def get_settings() -> Settings:
    return Settings()
