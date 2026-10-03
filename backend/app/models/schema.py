from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Turn(BaseModel):
    role: Literal["user", "ai"]
    text: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    text: str = Field(..., max_length=500, description="User question")
    style: Literal["short", "detailed"] = Field("short", description="Answer length the visitor chose")
    history: List[Turn] = Field(default_factory=list, max_length=8, description="Recent turns, oldest first, for follow-up questions")


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
