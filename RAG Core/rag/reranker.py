"""
reranker — Cross-encoder reranking on enriched_content.
"""

from typing import List

from rag.models import get_reranker
from rag.store import get_conn


def _get_texts(doc_keys: List[str]) -> dict[str, str]:
    """Fetch enriched_content for multiple doc_keys in a single query."""
    if not doc_keys:
        return {}
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT doc_key, enriched_content, page_content FROM chunks WHERE doc_key = ANY(%s)",
            (doc_keys,),
        )
        rows = cur.fetchall()
        result = {}
        for dk, ec, pc in rows:
            result[dk] = ec or pc or ""
        return result
    finally:
        cur.close()
        conn.close()


def rerank(query: str, candidates: List[str]) -> List[str]:
    """Score candidates with cross-encoder and return re-ranked doc_keys."""
    if not candidates:
        return []

    ce = get_reranker()
    texts = _get_texts(candidates)
    pairs = []
    for dk in candidates:
        text = texts.get(dk, "")
        pairs.append([query, text[:2048]])

    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]
