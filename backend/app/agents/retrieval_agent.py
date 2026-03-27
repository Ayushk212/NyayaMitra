"""
Agent 1: Retrieval Agent (Search Brain)
========================================
RESPONSIBILITY: Find relevant legal documents ONLY.
- Converts query → embedding
- Searches vector DB (semantic) + Whoosh (keyword)
- Returns top-K documents with metadata

MUST NOT: Summarize, answer, or interpret. Only retrieve.
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

from app.core.config import INDEX_DIR, TOP_K_RESULTS
from app.models.schemas import RetrievedChunk, RetrievalResult

# -- Whoosh (BM25 keyword search) --
from whoosh.index import open_dir, exists_in
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh import index as whoosh_index

# -- FAISS (vector search) --
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

WHOOSH_INDEX_DIR = INDEX_DIR / "whoosh"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNK_META_PATH = INDEX_DIR / "chunk_metadata.json"

# Schema for Whoosh
WHOOSH_SCHEMA = Schema(
    case_id=ID(stored=True),
    chunk_id=ID(stored=True, unique=True),
    case_name=STORED,
    court=STORED,
    date=STORED,
    citation=STORED,
    paragraph_ref=STORED,
    content=TEXT(stored=True),
)


def get_whoosh_index():
    """Get or return None for the Whoosh index."""
    if WHOOSH_INDEX_DIR.exists() and exists_in(str(WHOOSH_INDEX_DIR)):
        return open_dir(str(WHOOSH_INDEX_DIR))
    return None


def get_faiss_index():
    """Load the FAISS index from disk."""
    if not FAISS_AVAILABLE or not FAISS_INDEX_PATH.exists():
        return None
    return faiss.read_index(str(FAISS_INDEX_PATH))


def load_chunk_metadata() -> list[dict]:
    """Load chunk metadata mapping FAISS index positions to case info."""
    if not CHUNK_META_PATH.exists():
        return []
    with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def embed_query(query: str) -> Optional[np.ndarray]:
    """Embed query using Gemini embedding model."""
    from app.core.config import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        return np.array(result["embedding"], dtype="float32").reshape(1, -1)
    except Exception:
        return None


def bm25_search(
    query: str,
    allowed_case_ids: Optional[set[str]] = None,
    limit: int = TOP_K_RESULTS,
) -> list[RetrievedChunk]:
    """
    BM25 keyword search via Whoosh.
    Returns raw retrieval results — no interpretation.
    """
    ix = get_whoosh_index()
    if ix is None:
        return []

    parser = MultifieldParser(["content"], schema=ix.schema, group=OrGroup)
    q = parser.parse(query)

    results = []
    with ix.searcher() as searcher:
        hits = searcher.search(q, limit=limit * 3)
        for hit in hits:
            # Apply filters
            if allowed_case_ids is not None and hit["case_id"] not in allowed_case_ids:
                continue

            results.append(RetrievedChunk(
                case_id=hit["case_id"],
                case_name=hit.get("case_name", "Unknown"),
                court=hit.get("court", "Unknown"),
                date=hit.get("date"),
                citation=hit.get("citation"),
                paragraph=hit["content"][:500],
                paragraph_ref=hit.get("paragraph_ref"),
                score=round(hit.score, 4),
            ))
            if len(results) >= limit:
                break

    return results


async def vector_search(
    query: str,
    allowed_case_ids: Optional[set[str]] = None,
    limit: int = TOP_K_RESULTS,
) -> list[RetrievedChunk]:
    """
    Semantic vector search via FAISS.
    Returns raw retrieval results — no interpretation.
    """
    query_vec = await embed_query(query)
    if query_vec is None:
        return []

    faiss_idx = get_faiss_index()
    if faiss_idx is None:
        return []

    metadata = load_chunk_metadata()
    if not metadata:
        return []

    # Normalize for cosine similarity
    faiss.normalize_L2(query_vec)
    search_limit = limit * 5 if allowed_case_ids is not None else min(limit, faiss_idx.ntotal)
    scores, indices = faiss_idx.search(query_vec, min(search_limit, faiss_idx.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        if allowed_case_ids is not None and meta["case_id"] not in allowed_case_ids:
            continue
            
        results.append(RetrievedChunk(
            case_id=meta["case_id"],
            case_name=meta.get("case_name", "Unknown"),
            court=meta.get("court", "Unknown"),
            date=meta.get("date"),
            citation=meta.get("citation"),
            paragraph=meta.get("chunk_text", "")[:500],
            paragraph_ref=meta.get("paragraph_ref"),
            score=round(float(score), 4),
        ))
        if len(results) >= limit:
            break

    return results


def reciprocal_rank_fusion(
    bm25_results: list[RetrievedChunk],
    vector_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merge BM25 and vector results using Reciprocal Rank Fusion.
    Produces a single ranked list.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(bm25_results):
        key = f"{chunk.case_id}:{chunk.paragraph_ref or chunk.paragraph[:50]}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(vector_results):
        key = f"{chunk.case_id}:{chunk.paragraph_ref or chunk.paragraph[:50]}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        chunk.score = round(scores[key], 4)
        results.append(chunk)

    return results


async def retrieve(
    query: str,
    court: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    case_type: Optional[str] = None,
    sort_by: str = "relevance",
    limit: int = TOP_K_RESULTS,
) -> RetrievalResult:
    """
    Main retrieval function — Agent 1's only public interface.

    Runs hybrid search (BM25 + vector) and returns merged, ranked results.
    Does NOT summarize, interpret, or answer. ONLY retrieves.
    """
    # 1. Fetch allowed cases from DB if filters exist
    allowed_case_ids = None
    if any([court, date_from, date_to, case_type]):
        from sqlalchemy.future import select
        from app.db.database import async_session
        from app.db.models import Case
        
        async with async_session() as session:
            q = select(Case.id)
            if court:
                q = q.filter(Case.court.ilike(f"%{court}%"))
            if case_type:
                q = q.filter(Case.case_type == case_type)
            if date_from:
                q = q.filter(Case.date >= date_from)
            if date_to:
                q = q.filter(Case.date <= date_to)
            result = await session.execute(q)
            allowed_case_ids = set(result.scalars().all())

            if not allowed_case_ids:
                return RetrievalResult(query=query, results=[], total=0)

    # 2. Run both searches
    bm25_results = bm25_search(query, allowed_case_ids=allowed_case_ids, limit=limit * 3)
    vector_results = await vector_search(query, allowed_case_ids=allowed_case_ids, limit=limit * 3)

    # 3. Merge via reciprocal rank fusion
    if bm25_results and vector_results:
        merged = reciprocal_rank_fusion(bm25_results, vector_results)
    elif vector_results:
        merged = vector_results
    else:
        merged = bm25_results
        
    # 4. Sort
    if sort_by == "date":
        merged.sort(key=lambda x: x.date or "", reverse=True)

    return RetrievalResult(
        query=query,
        results=merged[:limit],
        total=len(merged),
    )
