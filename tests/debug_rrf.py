"""Debug with new RRF weights."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.retriever import hybrid_search, _expand_query
from rag.reranker import rerank
from rag.store import get_conn

q = "C'est quoi les frais de scolarite a IIT ?"
expanded = _expand_query(q)
print(f"Expanded: {expanded}")

hybrid = hybrid_search(q, top_k_dense=80, top_k_sparse=100, use_graph=False)
print(f"Hybrid total: {len(hybrid)}")

# Find fee doc position
fee_keys = [i for i, dk in enumerate(hybrid) if "frais" in dk.lower()]
print(f"Fee doc positions (after fix): {fee_keys[:10]}")

# Check top 30
conn = get_conn()
cur = conn.cursor()
for i, dk in enumerate(hybrid[:30]):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    title = (row[0][:60] if row else "N/A")
    if "frais" in title.lower() or "proc" in title.lower() or "inscri" in title.lower():
        print(f"  [{i}] fee doc: {title}")
cur.close()
conn.close()
