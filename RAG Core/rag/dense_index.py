"""
dense_index — PGVector dense embedding + search (incremental).

New chunks get embedded via E5-large and upserted into PostgreSQL.
Does NOT mark chunks as indexed — the pipeline handles that.
"""

from typing import List, Dict, Tuple
import numpy as np

from rag.models import get_dense_model
from rag.store import get_unindexed_chunks, upsert_embedding, get_conn


def embed_unindexed_chunks(batch_size: int = 64) -> int:
    """Embed all unindexed chunks and store embeddings in PGVector.
    Does NOT mark chunks as indexed — the pipeline handles that.
    Returns the number of chunks embedded.
    """
    model = get_dense_model()
    chunks = get_unindexed_chunks()
    if not chunks:
        print("[dense] No unindexed chunks found.")
        return 0

    print(f"[dense] Embedding {len(chunks)} unindexed chunks...")
    texts = [
        "passage: " + (c.get("enriched_content") or c.get("page_content", ""))[:2048].replace("\n", " ")
        for c in chunks
    ]
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    conn = get_conn()
    cur = conn.cursor()
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            "UPDATE chunks SET embedding = %s::vector WHERE doc_key = %s",
            (emb.tolist(), chunk["doc_key"]),
        )
    cur.close()
    conn.close()

    doc_keys = [c["doc_key"] for c in chunks]
    print(f"[dense] Embedded {len(doc_keys)} unindexed chunks.")
    return len(doc_keys)


def embed_missing_chunks(batch_size: int = 64) -> int:
    """Embed all chunks that lack embeddings (even if marked indexed).
    This is a recovery function for chunks that were marked indexed
    but never embedded due to the pipeline's 500-chunk limit.
    """
    model = get_dense_model()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT doc_key, enriched_content, page_content FROM chunks WHERE embedding IS NULL"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("[dense] No chunks missing embeddings.")
        return 0

    print(f"[dense] Embedding {len(rows)} chunks missing embeddings...")
    total = 0
    conn = get_conn()
    cur = conn.cursor()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = []
        doc_keys = []
        for dk, ec, pc in batch:
            text = ("passage: " + (ec or pc or "")[:2048]).replace("\n", " ")
            texts.append(text)
            doc_keys.append(dk)

        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

        for dk, emb in zip(doc_keys, embeddings):
            cur.execute(
                "UPDATE chunks SET embedding = %s::vector WHERE doc_key = %s",
                (emb.tolist(), dk),
            )
        total += len(batch)
        if (i // batch_size) % 5 == 0:
            conn.commit()
            print(f"[dense] Progress: {total}/{len(rows)} chunks embedded")

    conn.commit()
    cur.close()
    conn.close()
    print(f"[dense] Done: {total} chunks embedded.")
    return total


def dense_search(query: str, k: int = 150) -> List[Tuple[str, float]]:
    """Search PGVector with cosine similarity. Returns [(doc_key, score)]."""
    model = get_dense_model()
    q_emb = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # FIX #3: serialize once — q_emb is 1024 floats, tolist() is not free
    q_emb_list = q_emb.tolist()

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT doc_key, 1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (q_emb_list, q_emb_list, k))
        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        return results
    finally:
        cur.close()
        conn.close()
