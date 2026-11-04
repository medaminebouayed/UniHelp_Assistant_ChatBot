"""
ingest — CLI: Add new documents to the pipeline.

Usage:
    python -m rag.ingest --corpus data/synthetic/corpus_merged_split.json
    python -m rag.ingest --corpus path/to/new_chunks.json
"""

import json
import argparse
from pathlib import Path

from rag.store import ensure_schema, insert_chunk
from rag.config import PROJECT_ROOT


def _doc_key(meta: dict) -> str:
    site = (meta.get("site") or "").strip().lower()
    title = (meta.get("title") or "").strip().lower()
    section = (meta.get("section_title") or meta.get("header") or "").strip().lower()
    return f"{site}|{title}|{section}"


def ingest_corpus(corpus_path: str):
    path = Path(corpus_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[ingest] File not found: {path}")
        return

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    ensure_schema()
    total = len(chunks)
    inserted = 0
    skipped = 0

    for c in chunks:
        meta = c.get("metadata", {})
        dk = _doc_key(meta)
        ok = insert_chunk(
            doc_key=dk,
            page_content=c.get("page_content", ""),
            enriched_content=c.get("enriched_content", c.get("page_content", "")),
            bm25_text=c.get("bm25_text", ""),
            metadata=meta,
        )
        if ok:
            inserted += 1
        else:
            skipped += 1

    print(f"[ingest] {inserted} inserted, {skipped} skipped (duplicates), {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest chunks into PostgreSQL")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON")
    args = parser.parse_args()
    ingest_corpus(args.corpus)
