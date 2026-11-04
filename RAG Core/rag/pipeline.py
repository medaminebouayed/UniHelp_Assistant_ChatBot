"""
pipeline — Orchestrator for the full RAG lifecycle.

  index_unindexed() — Synthetic gen → Embed → LLM KGE → BM25 rebuild
  query()           — Full RAG pipeline: retrieve → fuse → rerank
"""

from typing import List, Dict, Optional

from rag.store import ensure_schema, count_chunks, mark_indexed_all_unindexed
from rag.dense_index import embed_unindexed_chunks
from rag.sparse_index import rebuild_index
from rag.graph_index import sync_unindexed_chunks
from rag.synthetic import generate_for_unindexed
from rag.retriever import hybrid_search
from rag.reranker import rerank
from rag.config import TOP_K_RERANK


def index_unindexed(
    use_graph: bool = True,
    gen_synthetic: bool = True,
    synth_limit: int = 100,
):
    """Process all unindexed chunks through the full pipeline.

    Steps:
      1. Generate synthetic variants for real chunks without them
      2. Embed all unindexed chunks into PGVector
      3. Extract entities via LLM KGE → Neo4j (if use_graph=True)
      4. Rebuild BM25 keyword index
      5. Mark all processed chunks as indexed
    """
    ensure_schema()

    stats = count_chunks()
    print(f"\n[pipeline] Before indexing: {stats}")

    if stats["unindexed"] == 0 and stats.get("no_synth", 0) == 0:
        print("[pipeline] Nothing to index.")
        return

    # ── Step 1: Incremental synthetic generation ───────────────────────
    if gen_synthetic:
        n_synth = generate_for_unindexed(limit=synth_limit)
        if n_synth > 0:
            print(f"[pipeline] Synth gen complete: {n_synth} new variants inserted.")
            # Re-count now that synthetic chunks were inserted
            stats = count_chunks()
            print(f"[pipeline] After synth gen: {stats}")

    # ── Step 2: Dense embedding ────────────────────────────────────────
    n_dense = embed_unindexed_chunks()
    if n_dense > 0:
        print(f"[pipeline] Dense embedding complete: {n_dense} chunks embedded.")

    # ── Step 3: LLM Knowledge Graph Extraction (Neo4j) ─────────────────
    if use_graph and n_dense > 0:
        n_graph = sync_unindexed_chunks()
        print(f"[pipeline] LLM KGE complete: {n_graph} chunks extracted to Neo4j.")

    # ── Step 4: BM25 rebuild ───────────────────────────────────────────
    if n_dense > 0:
        rebuild_index()
        print("[pipeline] BM25 index rebuilt.")

    # ── Step 5: Mark only embedded chunks as indexed ───────────────────
    if n_dense > 0:
        n_marked = mark_indexed_all_unindexed()
        if n_marked > 0:
            print(f"[pipeline] Marked {n_marked} chunks as indexed.")

    stats = count_chunks()
    print(f"[pipeline] After indexing: {stats}")


def query(
    question: str,
    top_k: int = TOP_K_RERANK,
    use_graph: bool = False,
) -> List[Dict]:
    """Full RAG query: hybrid search → rerank → return top chunks with text."""
    candidates = hybrid_search(question, use_graph=use_graph)
    if not candidates:
        return []

    # FIX #1: Smaller pool — RRF already gives good pre-ranking.
    # 50 cross-encoder pairs was the biggest latency bottleneck.
    pool_size = min(top_k * 2, 30)
    reranked = rerank(question, candidates[:pool_size])

    # Then take the final top_k
    reranked = reranked[:top_k]

    # FIX #2: Single batch SELECT instead of N sequential round-trips.
    # Previous code issued one query per doc_key (up to 10 DB calls).
    from rag.store import get_conn
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT doc_key, site, title, section_title, content_type,
                   page_content, enriched_content,
                   questions, keywords, url
            FROM chunks WHERE doc_key = ANY(%s)
        """, (reranked,))
        rows = {row[0]: row for row in cur.fetchall()}
        # Re-order to match reranker order
        results = []
        for dk in reranked:
            row = rows.get(dk)
            if row:
                results.append({
                    "doc_key": row[0],
                    "site": row[1],
                    "title": row[2],
                    "section_title": row[3],
                    "content_type": row[4],
                    "page_content": row[5],
                    "enriched_content": row[6],
                    "questions": row[7],
                    "keywords": row[8],
                    "url": row[9],
                })
        return results
    finally:
        cur.close()
        conn.close()
