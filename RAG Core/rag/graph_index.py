"""
graph_index — Neo4j entity extraction via LLM KGE for unindexed chunks.

Uses kge_engine.py's KGEEngine (LLM-based) instead of regex for
accurate entity extraction on new documents.

Does NOT mark chunks as indexed — the pipeline handles that.

Usage (via pipeline):
    python -m rag.index --use-graph
"""

from typing import List, Dict, Optional
from pathlib import Path

from rag.store import get_unindexed_chunks
from rag.config import PROJECT_ROOT


def _pg_row_to_kge_chunk(row: dict) -> dict:
    """Convert a PostgreSQL row to KGEEngine's chunk format."""
    meta = {
        "site": row.get("site", ""),
        "title": row.get("title", ""),
        "header": row.get("section_title", ""),
        "section_title": row.get("section_title", ""),
        "category": row.get("content_type", "general"),
        "content_type": row.get("content_type", "general"),
        "summary": row.get("summary", ""),
        "keywords": row.get("keywords", []) if isinstance(row.get("keywords"), list) else [],
        "url": row.get("url", ""),
        "is_synthetic": row.get("is_synthetic", False),
        "token_count": row.get("token_count", 0),
        "synthetic_id": row.get("synthetic_id", ""),
        "synthetic_style": row.get("synthetic_style", ""),
    }
    return {
        "page_content": row.get("page_content", "") or "",
        "enriched_content": row.get("enriched_content", "") or row.get("page_content", "") or "",
        "metadata": meta,
    }


def _extract_entities(query: str) -> list:
    """Extract meaningful search terms from a query (French-aware)."""
    import re
    FRENCH_STOP = {
        "le", "la", "les", "un", "une", "des", "du", "de", "ce", "cet", "cette",
        "ces", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses",
        "nos", "vos", "leurs", "notre", "votre", "leur",
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "qui", "que", "quoi", "dont", "ou", "et",
        "est", "sont", "dans", "sur", "avec", "pour", "par",
        "en", "au", "aux", "sous", "entre", "vers", "chez", "sans",
        "plus", "moins", "tres", "peu", "trop", "assez",
        "ca", "la", "a", "c", "d", "j", "l", "m", "n", "s", "t", "y",
        "qu", "quels", "quelle", "quelles", "quel",
        "comment", "pourquoi", "combien",
    }
    terms = re.split(r"[,\s;:!?()'\"]+", query.lower())
    return [t for t in terms if len(t) > 2 and t not in FRENCH_STOP]


def sync_unindexed_chunks(
    checkpoint_dir: Optional[str] = None,
    batch_callback=None,
) -> int:
    """Extract entities from unindexed chunks using LLM KGE, write to Neo4j.

    Does NOT mark chunks as indexed — pipeline handles that.

    Args:
        checkpoint_dir: Directory to store checkpoint files.
        batch_callback: Optional callback(processed, total).

    Returns:
        Number of chunks processed.
    """
    chunks = get_unindexed_chunks()
    if not chunks:
        print("[graph] No unindexed chunks to sync.")
        return 0

    print(f"[graph] Found {len(chunks)} unindexed chunks for LLM KGE extraction...")

    from utils.graph.kge_engine import KGEEngine

    engine = KGEEngine()
    try:
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else PROJECT_ROOT / "data" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / "kge_incremental_checkpoint.json")

        kge_chunks = [_pg_row_to_kge_chunk(c) for c in chunks]

        def doc_key_fn(chunk):
            m = chunk.get("metadata", {})
            return m.get("synthetic_id") or f"{m.get('site','')}__{m.get('title','')}__{m.get('section_title','')}__{hash(chunk.get('page_content',''))}"

        engine.process_chunks(
            chunks=kge_chunks,
            doc_key_fn=doc_key_fn,
            checkpoint_file=ckpt_path,
            batch_callback=batch_callback,
        )
    finally:
        engine.close()

    print(f"[graph] LLM KGE extraction complete for {len(chunks)} chunks.")
    return len(chunks)
