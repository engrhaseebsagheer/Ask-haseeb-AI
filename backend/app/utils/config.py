import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Ask Haseeb AI")
    ENV: str = os.getenv("ENV", "dev")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "")

    # Retrieval settings
    TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0"))

    # CORS
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

@lru_cache
def get_settings() -> Settings:
    return Settings()
