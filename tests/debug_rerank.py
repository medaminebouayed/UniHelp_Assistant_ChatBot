"""Debug reranker with new RRF weights."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.retriever import hybrid_search, _expand_query
from rag.reranker import rerank
from rag.store import get_conn

q = "C'est quoi les frais de scolarite a IIT ?"

hybrid = hybrid_search(q, top_k_dense=80, top_k_sparse=100, use_graph=False)
print(f"Hybrid total: {len(hybrid)}")

reranked = rerank(q, hybrid[:30])
print(f"Reranked: {len(reranked)}")

conn = get_conn()
cur = conn.cursor()
print("\nTop 15 reranked:")
for i, dk in enumerate(reranked[:15]):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    title = row[0] if row else "N/A"
    fee_marker = " <-- FEE DOC" if "frais" in title.lower() else ""
    print(f"  [{i}] {title[:70]}{fee_marker}")

# Also check top 25
print("\nFull reranked list:")
for i, dk in enumerate(reranked):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    title = row[0] if row else "N/A"
    if "frais" in title.lower():
        print(f"  [{i}] {title[:70]}")
cur.close()
conn.close()
