"""Search API router — orchestrates Agent 1 → Agent 3 pipeline for search."""
from fastapi import APIRouter, Query
from typing import Optional

from app.agents import retrieval_agent
from app.agents.ui_agent import format_search_results
from app.models.schemas import SearchResponse, SearchResultItem

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search_cases(
    q: str = Query(..., min_length=1, description="Search query"),
    court: Optional[str] = Query(None, description="Filter by court"),
    date_from: Optional[str] = Query(None, description="Filter start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter end date (YYYY-MM-DD)"),
    judge: Optional[str] = Query(None, description="Filter by judge name"),
    case_type: Optional[str] = Query(None, description="Filter by case type"),
    sort_by: str = Query("relevance", description="Sort by 'relevance' or 'date'"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Search endpoint. Pipeline:
    User Query → Agent 1 (Retrieval) → Agent 3 (Format) → Response

    Agent 2 is NOT involved in search — only in chat.
    """
    # Agent 1: Retrieve
    retrieval = await retrieval_agent.retrieve(
        query=q, court=court, date_from=date_from, date_to=date_to, 
        case_type=case_type, sort_by=sort_by, limit=limit
    )

    # Agent 3: Format for UI
    results = []
    for chunk in retrieval.results:
        results.append(SearchResultItem(
            id=chunk.case_id,
            title=chunk.case_name,
            court=chunk.court,
            date=chunk.date,
            citation=chunk.citation,
            case_type=None,
            snippet=chunk.paragraph[:300],
            score=chunk.score,
        ))

    return SearchResponse(
        results=results,
        total=retrieval.total,
        page=page,
        limit=limit,
    )
