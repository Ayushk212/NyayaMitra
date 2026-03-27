"""Cases API router — serves case detail and summary."""
import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_session
from app.db.models import Case
from app.agents.ui_agent import format_case_paragraphs
from app.agents.reasoning_agent import generate_summary
from app.models.schemas import CaseDetail, CaseSummaryResponse

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str, session: AsyncSession = Depends(get_session)):
    """Get full case details with formatted paragraphs."""
    result = await session.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Agent 3: Format paragraphs for the case viewer
    paragraphs = format_case_paragraphs(case.full_text)

    judges = []
    if case.judges:
        try:
            judges = json.loads(case.judges)
        except (json.JSONDecodeError, TypeError):
            judges = [j.strip() for j in case.judges.split(",")]

    return CaseDetail(
        id=case.id,
        title=case.title,
        court=case.court,
        date=case.date,
        judges=judges,
        case_type=case.case_type,
        citation=case.citation,
        full_text=case.full_text,
        paragraphs=paragraphs,
    )


@router.post("/{case_id}/summary", response_model=CaseSummaryResponse)
async def summarize_case(case_id: str, session: AsyncSession = Depends(get_session)):
    """
    Generate AI summary of a case.
    Uses Agent 2 (Reasoning) for summarization only.
    """
    result = await session.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Agent 2: Generate structured summary
    summary = await generate_summary(case.full_text, case.title)

    return CaseSummaryResponse(
        case_id=case.id,
        facts=summary.get("facts", ""),
        issues=summary.get("issues", ""),
        judgment=summary.get("judgment", ""),
        ratio_decidendi=summary.get("ratio_decidendi", ""),
    )
