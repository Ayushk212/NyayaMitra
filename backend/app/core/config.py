"""Core configuration for the NyayaMitra backend."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
DB_PATH = BASE_DIR / "nyayamitra.db"

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Search config
TOP_K_RESULTS = 10
CHUNK_SIZE = 800  # tokens per chunk
CHUNK_OVERLAP = 100

# Database
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
