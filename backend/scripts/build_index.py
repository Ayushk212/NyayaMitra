"""
Data ingestion and indexing script.
Loads case law JSON → chunks → builds Whoosh & FAISS indexes → stores in SQLite.
"""
import json
import sys
import os
import asyncio
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, GEMINI_API_KEY
from app.db.database import engine, async_session, Base
from app.db.models import Case, CaseChunk
from app.agents.retrieval_agent import (
    WHOOSH_SCHEMA, WHOOSH_INDEX_DIR,
    FAISS_INDEX_PATH, CHUNK_META_PATH,
)

# Whoosh
from whoosh import index as whoosh_index

# Optional: FAISS + embeddings
try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss-cpu not available. Skipping vector index.")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Split text into overlapping chunks with paragraph references."""
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.\s)', text)
    chunks = []
    current_chunk = ""
    current_para_start = 1
    para_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_idx += 1

        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "paragraph_ref": f"¶{current_para_start}-{para_idx - 1}",
            })
            # Overlap: keep last part
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 5:]) if len(words) > overlap // 5 else ""
            current_chunk = overlap_text + " " + para
            current_para_start = para_idx
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "paragraph_ref": f"¶{current_para_start}-{para_idx}",
        })

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using Gemini embedding model."""
    if not GEMINI_API_KEY:
        print("[WARN] No GEMINI_API_KEY. Skipping embeddings.")
        return []

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        embeddings = []
        # Batch in groups of 10
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            for text in batch:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                embeddings.append(result["embedding"])
            print(f"  Embedded {min(i + 10, len(texts))}/{len(texts)} chunks")

        return embeddings
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []


async def ingest():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("NyayaMitra — Data Ingestion Pipeline")
    print("=" * 60)

    # 1. Init DB
    print("\n[1/5] Initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Load cases
    print("[2/5] Loading case data...")
    seed_file = DATA_DIR / "seed_cases.json"
    if not seed_file.exists():
        print(f"[ERROR] Seed file not found: {seed_file}")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"  Found {len(cases)} cases")

    # 3. Process cases → chunks → store in DB
    print("[3/5] Chunking and storing in database...")
    all_chunks = []
    all_chunk_meta = []

    async with async_session() as session:
        for case_data in cases:
            # Store case
            case = Case(
                id=case_data["id"],
                title=case_data["title"],
                court=case_data["court"],
                date=case_data.get("date"),
                judges=json.dumps(case_data.get("judges", [])),
                case_type=case_data.get("case_type"),
                citation=case_data.get("citation"),
                full_text=case_data["full_text"],
            )
            session.add(case)

            # Chunk the text
            chunks = chunk_text(case_data["full_text"])
            for i, chunk in enumerate(chunks):
                db_chunk = CaseChunk(
                    case_id=case_data["id"],
                    chunk_text=chunk["text"],
                    chunk_index=i,
                    paragraph_ref=chunk["paragraph_ref"],
                    embedding_index=len(all_chunks),
                )
                session.add(db_chunk)

                all_chunks.append(chunk["text"])
                all_chunk_meta.append({
                    "case_id": case_data["id"],
                    "case_name": case_data["title"],
                    "court": case_data["court"],
                    "date": case_data.get("date"),
                    "citation": case_data.get("citation"),
                    "chunk_text": chunk["text"],
                    "paragraph_ref": chunk["paragraph_ref"],
                })

            print(f"  ✓ {case_data['title'][:50]}... ({len(chunks)} chunks)")

        await session.commit()

    print(f"  Total chunks: {len(all_chunks)}")

    # 4. Build Whoosh (BM25) index
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
    print(f"  ✓ Whoosh index created with {len(all_chunk_meta)} documents")

    # 5. Build FAISS vector index
    print("[5/5] Building FAISS vector index...")
    if FAISS_AVAILABLE:
        embeddings = embed_texts(all_chunks)
        if embeddings:
            dim = len(embeddings[0])
            faiss_idx = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

            vectors = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(vectors)
            faiss_idx.add(vectors)

            faiss.write_index(faiss_idx, str(FAISS_INDEX_PATH))
            print(f"  ✓ FAISS index created with {faiss_idx.ntotal} vectors (dim={dim})")
        else:
            print("  ⚠ No embeddings generated. FAISS index skipped.")
    else:
        print("  ⚠ FAISS not available. Vector search disabled.")

    # Save chunk metadata
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunk_meta, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Chunk metadata saved ({len(all_chunk_meta)} entries)")

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"  Cases: {len(cases)}")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Whoosh index: {WHOOSH_INDEX_DIR}")
    print(f"  FAISS index: {FAISS_INDEX_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(ingest())
