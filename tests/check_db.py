"""Check DB state for accreditation chunks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.store import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute("SELECT doc_key, site, title, indexed_at IS NOT NULL as is_indexed, embedding IS NOT NULL as has_embed FROM chunks WHERE LOWER(title) LIKE '%%accred%%' LIMIT 10")
rows = cur.fetchall()
print(f"Accreditation chunks found: {len(rows)}")
for r in rows:
    print(f"  key={str(r[0])[:60]}, site={r[1]}, title={r[2]}, indexed={r[3]}, has_embed={r[4]}")

cur.execute("SELECT COUNT(*) FROM chunks")
print(f"Total chunks: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM chunks WHERE indexed_at IS NOT NULL")
print(f"Indexed chunks: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
print(f"Chunks with embedding: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM chunks WHERE LOWER(title) LIKE '%%formation%%'")
print(f"FORMATION chunks: {cur.fetchone()[0]}")

cur.execute("SELECT title, COUNT(*) FROM chunks WHERE LOWER(title) LIKE '%%formation%%' GROUP BY title LIMIT 5")
for r in cur.fetchall():
    print(f"  title='{r[0]}' count={r[1]}")

cur.close()
conn.close()
