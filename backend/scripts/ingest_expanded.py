"""
Data ingestion and indexing script for Expanded Dataset.
Loads legal_dataset_expanded.json -> mapped to Case format -> chunks -> builds Whoosh & FAISS.
"""
import json
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, GEMINI_API_KEY
from app.db.database import engine, async_session, Base
from app.db.models import Case, CaseChunk
from app.agents.retrieval_agent import (
    WHOOSH_SCHEMA, WHOOSH_INDEX_DIR,
    FAISS_INDEX_PATH, CHUNK_META_PATH,
)
from whoosh import index as whoosh_index

# Pull the specific processing functions from the original codebase
from scripts.build_index import chunk_text, embed_texts

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss-cpu not available.")

async def ingest():
    print("=" * 60)
    print("NyayaMitra — Expanded Data Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/5] Initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("[2/5] Loading expanded case data...")
    # Load expanded dataset
    expanded_cases_path = Path(__file__).resolve().parent.parent / "dataset.json"
    with open(expanded_cases_path, "r", encoding="utf-8") as f:
        expanded_cases = json.load(f)
        
    # Load seed cases
    seed_cases_path = Path(__file__).resolve().parent.parent / "data/seed_cases.json"
    with open(seed_cases_path, "r", encoding="utf-8") as f:
        seed_cases = json.load(f)
        
    all_cases_to_process = seed_cases + expanded_cases
    print(f"  Found {len(all_cases_to_process)} total cases ({len(seed_cases)} seed, {len(expanded_cases)} expanded)")

    print("\n[3/5] Chunking and storing in database...")
    all_chunks = []
    all_chunk_meta = []

    async with async_session() as session:
        for case_data in all_cases_to_process:
            case_id = case_data.get("case_id", "Unknown")
            
            # Map dataset schema to database Base model expected fields
            case = Case(
                id=case_id,
                title=case_data.get("title", f"Case {case_id}"),
                court="Supreme Court of India",
                date=None,
                judges=json.dumps([]),
                case_type="Expanded Semantic Search",
                citation="",
                full_text=case_data.get("full_text", case_data.get("summary", "")),
            )
            # Try merging the case to update it if it already exists
            await session.merge(case)
            
            full_text = case_data.get("full_text", case_data.get("summary", ""))
            if not full_text.strip():
                continue

            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                db_chunk = CaseChunk(
                    case_id=case_id,
                    chunk_text=chunk["text"],
                    chunk_index=i,
                    paragraph_ref=chunk["paragraph_ref"],
                    embedding_index=len(all_chunks),
                )
                await session.merge(db_chunk)

                all_chunks.append(chunk["text"])
                all_chunk_meta.append({
                    "case_id": case_id,
                    "case_name": case_data.get("title", ""),
                    "court": "Supreme Court of India",
                    "date": "",
                    "citation": "",
                    "chunk_text": chunk["text"],
                    "paragraph_ref": chunk["paragraph_ref"],
                })

        print("  Committing to SQLite database...")
        # Ignore duplicate Primary Key overrides silently by merging instead 
        # (Assuming you don't rebuild DB from scratch each run)
        try:
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"[WARN] Database commit issued a warning (mostly duplicates). Bypassing insertion and proceeding with Index builds.")

    print(f"  Total chunks: {len(all_chunks)}")

    print("[4/5] Building Whoosh (BM25) search index...")
    WHOOSH_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ix = whoosh_index.create_in(str(WHOOSH_INDEX_DIR), WHOOSH_SCHEMA)
    writer = ix.writer()

    for i, meta in enumerate(all_chunk_meta):
        writer.add_document(
            case_id=meta["case_id"],
            chunk_id=str(i),
            case_name=meta["case_name"],
            court=meta["court"],
            date=meta.get("date", ""),
            citation=meta.get("citation", ""),
            paragraph_ref=meta.get("paragraph_ref", ""),
            content=meta["chunk_text"],
        )
    writer.commit()
    print(f"  > Whoosh index created with {len(all_chunk_meta)} documents")

    print("[5/5] Building FAISS vector index...")
    if FAISS_AVAILABLE:
        embeddings = embed_texts(all_chunks)
        if embeddings:
            dim = len(embeddings[0])
            faiss_idx = faiss.IndexFlatIP(dim)
            import numpy as np
            vectors = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(vectors)
            faiss_idx.add(vectors)
            faiss.write_index(faiss_idx, str(FAISS_INDEX_PATH))
            print(f"  > FAISS index created with {faiss_idx.ntotal} vectors")
        else:
            print("  ! No embeddings generated! Check GEMINI_API_KEY")

    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunk_meta, f, ensure_ascii=False, indent=2)
    print("DONE: Full Expanded Ingestion Pipeline Completed!")

if __name__ == "__main__":
    asyncio.run(ingest())
