"""
store — PostgreSQL chunk persistence with incremental indexing tracking.

Schema:
  chunks table (PGVector compatible):
    id SERIAL PRIMARY KEY
    doc_key TEXT UNIQUE NOT NULL
    site, title, section_title, content_type, ...
    page_content, enriched_content, bm25_text
    questions JSONB, keywords JSONB, qa_pairs JSONB
    embedding vector(1024)
    indexed_at TIMESTAMP       -- NULL = not yet indexed
    created_at TIMESTAMP
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import psycopg2
import psycopg2.extras

from rag.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, EMBED_DIM


def get_conn():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD,
    )
    conn.autocommit = True
    return conn


def ensure_schema():
    """Create PGVector extension + chunks table if not exists."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id              SERIAL PRIMARY KEY,
            doc_key         TEXT UNIQUE NOT NULL,
            site            TEXT,
            title           TEXT,
            section_title   TEXT,
            content_type    TEXT,
            page_content    TEXT,
            enriched_content TEXT,
            bm25_text       TEXT,
            questions       JSONB DEFAULT '[]',
            keywords        JSONB DEFAULT '[]',
            qa_pairs        JSONB DEFAULT '[]',
            summary         TEXT,
            url             TEXT,
            has_noise           BOOLEAN DEFAULT FALSE,
            is_synthetic        BOOLEAN DEFAULT FALSE,
            synthetic_style     TEXT DEFAULT '',
            synth_generated_at  TIMESTAMP,
            token_count         INT,
            chunk_length        INT,
            embedding           vector({EMBED_DIM}),
            indexed_at          TIMESTAMP,
            created_at          TIMESTAMP DEFAULT NOW()
        )
    """)
    # Add indexed_at column if it doesn't exist (migration for existing DBs)
    try:
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP")
    except Exception:
        pass
    # Add created_at column if it doesn't exist
    try:
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()")
    except Exception:
        pass
    # Migration for file storage (BYTEA blobs)
    for col_def in [
        "ADD COLUMN IF NOT EXISTS file_data BYTEA",
        "ADD COLUMN IF NOT EXISTS file_name TEXT DEFAULT ''",
        "ADD COLUMN IF NOT EXISTS file_type TEXT DEFAULT ''",
    ]:
        try:
            cur.execute(f"ALTER TABLE chunks {col_def}")
        except Exception:
            pass
    # Migration for existing DBs that lack these columns
    for col in ["synth_generated_at", "is_synthetic", "synthetic_style"]:
        try:
            cur.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} TIMESTAMP" if col == "synth_generated_at"
                        else f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} BOOLEAN DEFAULT FALSE" if col == "is_synthetic"
                        else f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} TEXT DEFAULT ''")
        except Exception:
            pass
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_indexed_at ON chunks (indexed_at)")
    cur.close()
    conn.close()


def insert_chunk(
    doc_key: str,
    page_content: str,
    enriched_content: str,
    bm25_text: str,
    metadata: Dict,
) -> bool:
    """Insert a new chunk. Returns True if inserted, False if doc_key exists."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chunks (
                doc_key, site, title, section_title, content_type,
                page_content, enriched_content, bm25_text,
                questions, keywords, qa_pairs, summary, url,
                has_noise, is_synthetic, synthetic_style,
                token_count, chunk_length
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_key) DO NOTHING
        """, (
            doc_key,
            metadata.get("site", ""),
            metadata.get("title", ""),
            metadata.get("section_title", "") or metadata.get("header", ""),
            metadata.get("content_type", "general"),
            page_content,
            enriched_content or page_content,
            bm25_text or page_content,
            json.dumps(metadata.get("questions", []), ensure_ascii=False),
            json.dumps(metadata.get("keywords", []), ensure_ascii=False),
            json.dumps(metadata.get("qa_pairs", []), ensure_ascii=False),
            metadata.get("summary", ""),
            metadata.get("url", ""),
            bool(metadata.get("has_noise", False)),
            bool(metadata.get("is_synthetic", False)),
            metadata.get("synthetic_style", ""),
            metadata.get("token_count"),
            metadata.get("chunk_length") or len(page_content),
        ))
        return cur.rowcount > 0
    finally:
        cur.close()
        conn.close()


def get_unindexed_chunks(limit: int = 500) -> List[Dict]:
    """Get chunks where indexed_at is NULL (not yet embedded/entity-extracted)."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM chunks
        WHERE indexed_at IS NULL
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_all_chunks() -> List[Dict]:
    """Get ALL chunks (for BM25 full rebuild)."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM chunks ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def mark_indexed(doc_keys: List[str]):
    """Set indexed_at = NOW() for the given doc_keys."""
    if not doc_keys:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE chunks SET indexed_at = NOW()
        WHERE doc_key = ANY(%s)
    """, (doc_keys,))
    cur.close()
    conn.close()


def upsert_embedding(doc_key: str, embedding: List[float]):
    """Update embedding for an existing chunk (upsert by doc_key)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE chunks SET embedding = %s::vector
        WHERE doc_key = %s
    """, (embedding, doc_key))
    cur.close()
    conn.close()


def count_chunks() -> Dict:
    """Get chunk counts (total, indexed, unindexed)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE indexed_at IS NOT NULL) AS indexed,
            COUNT(*) FILTER (WHERE indexed_at IS NULL) AS unindexed
        FROM chunks
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return {"total": row[0], "indexed": row[1], "unindexed": row[2]}


def get_real_chunks_without_synthetic(limit: int = 100) -> List[Dict]:
    """Get real chunks that don't have synthetic variants generated yet."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM chunks
        WHERE indexed_at IS NULL
          AND (is_synthetic IS NULL OR is_synthetic = FALSE)
          AND synth_generated_at IS NULL
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def mark_synthetic_generated(doc_keys: List[str]):
    """Set synth_generated_at = NOW() for the given doc_keys."""
    if not doc_keys:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE chunks SET synth_generated_at = NOW()
        WHERE doc_key = ANY(%s)
    """, (doc_keys,))
    cur.close()
    conn.close()


def mark_indexed_all_unindexed():
    """Mark all NULL indexed_at chunks as indexed NOW()."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE chunks SET indexed_at = NOW() WHERE indexed_at IS NULL")
    n = cur.rowcount
    cur.close()
    conn.close()
    return n


def delete_chunk(doc_key: str) -> bool:
    """Delete a chunk by doc_key."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE doc_key = %s", (doc_key,))
    deleted = cur.rowcount > 0
    cur.close()
    conn.close()
    return deleted


def delete_chunks_by_document(site: str, title: str) -> int:
    """Delete all chunks matching a document's site and title (case-insensitive)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM chunks WHERE LOWER(site) = LOWER(%s) AND LOWER(title) = LOWER(%s)",
        (site, title),
    )
    n = cur.rowcount
    cur.close()
    conn.close()
    return n


def delete_chunks_by_filepath(filepath: str) -> int:
    """Delete all chunks whose url matches the given file path."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM chunks WHERE url = %s",
        (filepath,),
    )
    n = cur.rowcount
    cur.close()
    conn.close()
    return n
