"""
Config — All settings from .env with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── HF token ────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

# ── PostgreSQL ───────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST", "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5433"))
PG_DATABASE = os.getenv("PG_DATABASE", "graphrag")
PG_USER     = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password1234")

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# ── Models ───────────────────────────────────────────────────────────────────
DENSE_MODEL_NAME  = os.getenv("DENSE_MODEL", "intfloat/multilingual-e5-large")
RERANKER_NAME     = os.getenv("RERANKER_NAME", "BAAI/bge-reranker-v2-m3")
EMBED_DIM         = int(os.getenv("EMBED_DIM", "1024"))  # E5-large is 1024

# ── Paths ────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", str(PROJECT_ROOT / "data" / "bm25_index.pkl")))
CORPUS_MERGED   = PROJECT_ROOT / "data" / "synthetic" / "corpus_merged_split.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
RRF_K           = 60
DENSE_WEIGHT    = 0.35
SPARSE_WEIGHT   = 0.65
TOP_K_DENSE     = 40
TOP_K_SPARSE    = 60
TOP_K_RERANK    = 10
