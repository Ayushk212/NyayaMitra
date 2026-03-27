"""Chat API router — orchestrates the full 3-agent pipeline."""
import json
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.agents import retrieval_agent, reasoning_agent
from app.agents.ui_agent import format_answer_blocks
from app.models.schemas import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat(request: ChatRequest):
    """
    Full RAG pipeline:
    User Query → Agent 1 (Retrieve) → Agent 2 (Reason) → Agent 3 (Format) → User

    This is the CRITICAL path. Each agent does ONE job.
    """

    async def event_generator():
        # Step 1: Signal retrieval start
        yield {
            "event": "status",
            "data": json.dumps({"stage": "retrieving", "message": "Searching case law database..."})
        }

        # Agent 1: Retrieve relevant documents
        retrieval = await retrieval_agent.retrieve(
            query=request.query,
            limit=7,
        )

        yield {
            "event": "status",
            "data": json.dumps({
                "stage": "retrieved",
                "message": f"Found {len(retrieval.results)} relevant cases",
                "count": len(retrieval.results),
            })
        }

        # Send retrieved sources preview
        sources = []
        for chunk in retrieval.results[:5]:
            sources.append({
                "case_name": chunk.case_name,
                "court": chunk.court,
                "score": chunk.score,
            })
        yield {
            "event": "sources",
            "data": json.dumps(sources)
        }

        # Step 2: Signal reasoning start
        yield {
            "event": "status",
            "data": json.dumps({"stage": "reasoning", "message": "Analyzing legal principles..."})
        }

        # Agent 2: Generate reasoned answer from retrieved data ONLY
        reasoning = await reasoning_agent.reason(
            query=request.query,
            retrieved_chunks=retrieval.results,
            mode=request.mode,
        )

        # Step 3: Agent 3 formats the output
        formatted = format_answer_blocks(reasoning)

        # Stream the answer in chunks (simulate typing effect)
        answer = formatted["formatted_answer"]
        chunk_size = 15  # characters per chunk
        for i in range(0, len(answer), chunk_size):
            yield {
                "event": "answer_chunk",
                "data": json.dumps({"text": answer[i:i + chunk_size]})
            }
            await asyncio.sleep(0.02)  # typing effect

        # Send final structured data
        yield {
            "event": "complete",
            "data": json.dumps({
                "citations": formatted["citations_panel"],
                "confidence": formatted["confidence"],
                "suggestions": formatted["suggestions"],
                "mode": formatted["mode"],
            })
        }

    return EventSourceResponse(event_generator())


@router.post("/quick")
async def quick_chat(request: ChatRequest):
    """
    Non-streaming version for simple requests.
    Same 3-agent pipeline, returns complete response.
    """
    # Agent 1: Retrieve
    retrieval = await retrieval_agent.retrieve(query=request.query, limit=7)

    # Agent 2: Reason
    reasoning = await reasoning_agent.reason(
        query=request.query,
        retrieved_chunks=retrieval.results,
        mode=request.mode,
    )

    # Agent 3: Format
    formatted = format_answer_blocks(reasoning)

    return formatted
