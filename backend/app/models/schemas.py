"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel
from typing import Optional


class SearchRequest(BaseModel):
    query: str
    court: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    judge: Optional[str] = None
    case_type: Optional[str] = None
    page: int = 1
    limit: int = 10


class RetrievedChunk(BaseModel):
    case_id: str
    case_name: str
    court: str
    date: Optional[str]
    citation: Optional[str]
    paragraph: str
    paragraph_ref: Optional[str]
    score: float


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total: int


class Citation(BaseModel):
    case_name: str
    paragraph: str
    paragraph_ref: Optional[str] = None


class ReasoningResult(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    mode: str = "formal"


class ChatRequest(BaseModel):
    query: str
    mode: str = "formal"  # "formal" or "eli5"


class CaseDetail(BaseModel):
    id: str
    title: str
    court: str
    date: Optional[str]
    judges: Optional[list[str]]
    case_type: Optional[str]
    citation: Optional[str]
    full_text: str
    paragraphs: list[dict]


class CaseSummaryResponse(BaseModel):
    case_id: str
    facts: str
    issues: str
    judgment: str
    ratio_decidendi: str


class SearchResultItem(BaseModel):
    id: str
    title: str
    court: str
    date: Optional[str]
    citation: Optional[str]
    case_type: Optional[str]
    snippet: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    page: int
    limit: int
