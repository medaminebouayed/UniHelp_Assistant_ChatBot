"""Direct debug: check why fee doc isn't retrieved."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.retriever import hybrid_search, dense_search, sparse_search, _expand_query
from rag.reranker import rerank

q = "C'est quoi les frais de scolarite a IIT ?"

# Check dense search
print("=== DENSE SEARCH (top 80) ===")
d = dense_search(q, k=80)
print(f"Dense results count: {len(d)}")
for i, (dk, score) in enumerate(d):
    if "frais" in dk.lower() or "procéd" in dk.lower() or "procedure" in dk.lower():
        print(f"  [{i}] {dk} (score={score:.4f})")

# Check expanded query
expanded = _expand_query(q)
print(f"\nExpanded query: {expanded}")

# Check sparse search
print(f"\n=== SPARSE SEARCH (top 100) ===")
s = sparse_search(expanded, k=100)
print(f"Sparse results count: {len(s)}")
for i, (dk, score) in enumerate(s):
    if "frais" in dk.lower() or "procéd" in dk.lower() or "procedure" in dk.lower():
        print(f"  [{i}] {dk} (score={score:.4f})")

# Check hybrid search
print(f"\n=== HYBRID SEARCH ===")
h = hybrid_search(q, top_k_dense=80, top_k_sparse=100, use_graph=True)
print(f"Hybrid results count: {len(h)}")
for i, dk in enumerate(h):
    if "frais" in dk.lower() or "procéd" in dk.lower():
        print(f"  [{i}] {dk}")

# Check if it's in the top 50
for i, dk in enumerate(h[:50]):
    if "frais" in dk.lower() or "procéd" in dk.lower():
        print(f"  TOP50 [{i}] {dk}")

if h:
    # Rerank top 25
    print(f"\n=== RERANKER (top 25) ===")
    r = rerank(q, h[:25])
    print(f"Reranked count: {len(r)}")
    from rag.store import get_conn
    conn = get_conn()
    cur = conn.cursor()
    for i, dk in enumerate(r):
        cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
        row = cur.fetchone()
        title = row[0] if row else "N/A"
        if "frais" in title.lower() or "procéd" in title.lower():
            print(f"  [{i}] {dk} -> {title}")
    cur.close()
    conn.close()
