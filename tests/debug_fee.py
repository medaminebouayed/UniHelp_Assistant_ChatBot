"""Debug fee doc retrieval step by step."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.retriever import hybrid_search, _expand_query
from rag.reranker import rerank
from rag.store import get_conn
from rag.graph_index import _extract_entities
from rag.retriever import _accent_fold

q = "C'est quoi les frais de scolarite a IIT ?"

# Step 1: Check with augmented query
from rag.retriever import _expand_query

# Check expansion
expanded = _expand_query(q)
print(f"Original: {q}")
print(f"Expanded: {expanded}")

# Step 2: Hybrid search
hybrid = hybrid_search(q, top_k_dense=80, top_k_sparse=100, use_graph=False)
print(f"\nHybrid results: {len(hybrid)}")

# Find fee position
conn = get_conn()
cur = conn.cursor()
for i, dk in enumerate(hybrid):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    if row and "frais" in row[0].lower():
        print(f"  Fee doc at hybrid[{i}]: {row[0]}")

# Step 3: Reranker 
reranked = rerank(q, hybrid[:50])
print(f"\nReranked results: {len(reranked)}")
for i, dk in enumerate(reranked):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    if row and "frais" in row[0].lower():
        print(f"  Fee doc at reranked[{i}]: {row[0]}")

cur.close()
conn.close()
