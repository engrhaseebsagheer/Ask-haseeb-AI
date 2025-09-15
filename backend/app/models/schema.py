from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    text: str = Field(..., description="User question")

class MatchChunk(BaseModel):
    score: float
    title: Optional[str] = None
    source: Optional[str] = None
    text: Optional[str] = None

class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved: List[MatchChunk] = []

class HealthResponse(BaseModel):
    status: str
    app: str
