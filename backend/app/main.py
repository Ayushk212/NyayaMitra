"""FastAPI main application — NyayaMitra Legal AI Platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import FRONTEND_URL
from app.db.database import init_db
from app.routers import search, cases, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init database
    await init_db()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="NyayaMitra API",
    description="Legal AI Platform for Indian Case Law — 3-Agent Architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(search.router)
app.include_router(cases.router)
app.include_router(chat.router)


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "NyayaMitra API"}


@app.get("/api/trending")
async def trending_queries():
    """Return trending/popular search queries for the home page."""
    return {
        "queries": [
            {"text": "Right to Privacy", "category": "Constitutional"},
            {"text": "Bail Conditions in Non-Bailable Offenses", "category": "Criminal"},
            {"text": "Article 21 Right to Life", "category": "Fundamental Rights"},
            {"text": "Dowry Death Section 304B", "category": "Criminal"},
            {"text": "Arbitration Agreement Validity", "category": "Civil"},
            {"text": "Environmental Protection PIL", "category": "Environmental"},
        ]
    }
