"""
Fix missing embeddings in the database.

Only 1071/5291 chunks have embeddings because the old pipeline
marked ALL chunks as indexed before embedding them all.
This script re-embeds all chunks without embeddings, then rebuilds the BM25 index.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.store import get_conn
from rag.dense_index import embed_missing_chunks
from rag.sparse_index import rebuild_index
from rag.pipeline import mark_indexed_all_unindexed

print("=" * 60)
print("FIX: Embedding all chunks without embeddings")
print("=" * 60)

# Step 1: Check how many need embedding
conn = get_conn()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
missing = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
have = cur.fetchone()[0]
print(f"Chunks with embeddings: {have}")
print(f"Chunks missing embeddings: {missing}")

# Step 2: Reset indexed_at for chunks without embeddings
# so get_unindexed_chunks will find them in future pipeline runs
cur.execute("UPDATE chunks SET indexed_at = NULL WHERE embedding IS NULL")
reset_count = cur.rowcount
conn.commit()
print(f"Reset indexed_at for {reset_count} chunks")
cur.close()
conn.close()

# Step 3: Embed all missing chunks
if missing > 0:
    print(f"\nEmbedding {missing} chunks (this may take a while)...")
    n = embed_missing_chunks(batch_size=64)
    print(f"Embedded {n} chunks")

    # Step 4: Mark newly embedded chunks as indexed
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE chunks SET indexed_at = NOW() WHERE indexed_at IS NULL AND embedding IS NOT NULL")
    marked = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"Marked {marked} newly embedded chunks as indexed")

    # Step 5: Rebuild BM25 index
    print("\nRebuilding BM25 index...")
    rebuild_index()
    print("BM25 index rebuilt.")
else:
    print("No chunks need embedding. DB is healthy.")

# Step 6: Final counts
conn = get_conn()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
still_missing = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
have_now = cur.fetchone()[0]
cur.close()
conn.close()
print(f"\nFinal state:")
print(f"  With embeddings: {have_now}")
print(f"  Still missing: {still_missing}")
print("\nDone! Restart the RAG API for changes to take effect.")
