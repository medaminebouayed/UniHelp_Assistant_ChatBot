"""
embed_to_postgreSQL.py — Embed corpus chunks into PostgreSQL (PGVector)
=======================================================================
Stores chunk embeddings + metadata for hybrid vector+keyword retrieval.

Uses:
  - sentence-transformers/all-MiniLM-L6-v2  (384-dim embeddings)
  - PGVector extension for PostgreSQL        (cosine similarity)
  - enriched_content field for vector search
  - bm25_text field stored alongside for BM25 compatibility

Dependencies:
    pip install sentence-transformers psycopg2-binary sqlalchemy pgvector

Usage:
    py utils/embed_to_postgreSQL.py --corpus data/synthetic/corpus_final.json
    py utils/embed_to_postgreSQL.py --corpus data/synthetic/corpus_final.json --rebuild
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_HF_TOKEN = os.getenv("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── PostgreSQL connection defaults (override via .env) ──────────────────
PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5433"))
PG_DATABASE = os.getenv("PG_DATABASE", "graphrag")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password1234")

# ── Embedding model ─────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM        = 384

# ── Batch size for inserts ──────────────────────────────────────────────
BATCH_SIZE = 64


def get_connection():
    """Create a psycopg2 connection with PGVector support."""
    import psycopg2
    conn = psycopg2.connect(
        host     = PG_HOST,
        port     = PG_PORT,
        dbname   = PG_DATABASE,
        user     = PG_USER,
        password = PG_PASSWORD,
    )
    conn.autocommit = True
    return conn


def ensure_schema(conn):
    """Create PGVector extension and chunks table if they don't exist."""
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id            SERIAL PRIMARY KEY,
            doc_key       TEXT UNIQUE NOT NULL,
            site          TEXT,
            title         TEXT,
            category      TEXT,
            header        TEXT,
            content_type  TEXT,
            page_content  TEXT,
            enriched_content TEXT,
            bm25_text     TEXT,
            section_title TEXT,
            summary       TEXT,
            questions     JSONB DEFAULT '[]',
            keywords      JSONB DEFAULT '[]',
            qa_pairs      JSONB DEFAULT '[]',
            has_noise     BOOLEAN DEFAULT FALSE,
            retrieval_boost FLOAT DEFAULT 1.0,
            filter_sim_score FLOAT,
            factual_recall  FLOAT,
            token_count   INT,
            chunk_length  INT,
            url           TEXT,
            embedding     vector(%d)
        )
    """ % EMBED_DIM)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """ % ())
    cur.close()


def load_corpus(corpus_path: str) -> list:
    path = Path(corpus_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / corpus_path
    if not path.exists():
        print(f"[error] Corpus not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[info] Loaded {len(data)} chunks from {path}")
    return data


def chunk_generator(chunks: list, batch_size: int = BATCH_SIZE):
    """Yield batches of chunks for embedding + insert."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


def embed_batch(model: SentenceTransformer, batch: list, field: str) -> np.ndarray:
    """Embed a batch of chunks using the specified text field."""
    texts = [chunk.get(field, chunk.get("page_content", "")) for chunk in batch]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _safe(val, default=""):
    """Return str(val) if val is a dict/list, otherwise val as-is."""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return val if val is not None else default

def _safe_float(val, default=None):
    """Safely convert to float, handling dicts/lists."""
    if val is None or isinstance(val, (dict, list)):
        return default
    return float(val)

def _doc_key(meta: dict) -> str:
    site = (meta.get("site") or "").strip().lower()
    title = (meta.get("title") or "").strip().lower()
    section = (meta.get("section_title") or meta.get("header") or "").strip().lower()
    return f"{site}|{title}|{section}"

def insert_batch(conn, batch: list, embeddings: np.ndarray):
    """Insert or update a batch of chunks with embeddings."""
    cur = conn.cursor()
    values = []
    for chunk, emb in zip(batch, embeddings):
        meta = chunk.get("metadata", {})
        doc_key = _doc_key(meta)

        values.append((
            doc_key,
            _safe(meta.get("site")),
            _safe(meta.get("title")),
            _safe(meta.get("category")),
            _safe(meta.get("header")),
            _safe(meta.get("content_type")),
            _safe(chunk.get("page_content"), ""),
            _safe(chunk.get("enriched_content"), _safe(chunk.get("page_content"), "")),
            _safe(chunk.get("bm25_text"), ""),
            _safe(meta.get("section_title", "")),
            _safe(meta.get("summary", "")),
            json.dumps(meta.get("questions", []), ensure_ascii=False),
            json.dumps(meta.get("keywords", []), ensure_ascii=False),
            json.dumps(meta.get("qa_pairs", []), ensure_ascii=False),
            bool(_safe_float(meta.get("has_noise"), 0)),
            _safe_float(meta.get("retrieval_boost"), 1.0),
            _safe_float(meta.get("filter_sim_score")),
            _safe_float(meta.get("factual_recall")),
            _safe_float(meta.get("token_count")),
            _safe_float(meta.get("chunk_length")),
            _safe(meta.get("url", "")),
            emb.tolist(),
        ))

    cur.executemany("""
        INSERT INTO chunks (
            doc_key, site, title, category, header, content_type,
            page_content, enriched_content, bm25_text,
            section_title, summary, questions, keywords, qa_pairs,
            has_noise, retrieval_boost, filter_sim_score, factual_recall,
            token_count, chunk_length, url, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (doc_key) DO UPDATE SET
            embedding         = EXCLUDED.embedding,
            enriched_content  = EXCLUDED.enriched_content,
            bm25_text         = EXCLUDED.bm25_text,
            questions         = EXCLUDED.questions,
            keywords          = EXCLUDED.keywords
    """, values)
    cur.close()


def run_embedding(corpus_path: str, rebuild: bool = False, limit: int = None):
    print(f"\n[embed] Loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
    print(f"[embed] Model dimension: {dim}")

    chunks = load_corpus(corpus_path)
    if not chunks:
        return

    if limit:
        chunks = chunks[:limit]
        print(f"[embed] Limited to {limit} chunks")

    conn = get_connection()
    ensure_schema(conn)

    if rebuild:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE")
        cur.close()
        print("[embed] Table truncated for rebuild")

    total = len(chunks)
    print(f"[embed] Embedding and inserting {total} chunks in batches of {BATCH_SIZE}...")

    for batch in tqdm(chunk_generator(chunks), total=(total + BATCH_SIZE - 1) // BATCH_SIZE):
        embeddings = embed_batch(model, batch, "enriched_content")
        insert_batch(conn, batch, embeddings)

    conn.close()
    print(f"[embed] Done — {total} chunks stored in PostgreSQL (PGVector)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks into PostgreSQL (PGVector)")
    parser.add_argument("--corpus", default="data/synthetic/corpus_final.json",
                        help="Path to merged corpus JSON")
    parser.add_argument("--rebuild", action="store_true",
                        help="Truncate table before inserting")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N chunks (for testing)")
    args = parser.parse_args()

    run_embedding(args.corpus, rebuild=args.rebuild, limit=args.limit)
