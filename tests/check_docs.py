"""Check which documents are in the DB and their titles."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))
from rag.store import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute("SELECT DISTINCT title FROM chunks WHERE LOWER(title) LIKE '%frais%'")
print("Fee-related documents:")
for r in cur.fetchall():
    print(f"  - {r[0]}")

cur.execute("SELECT DISTINCT title FROM chunks WHERE LOWER(title) LIKE '%accredit%'")
print("\nAccreditation-related documents:")
for r in cur.fetchall():
    print(f"  - {r[0]}")

cur.execute("SELECT DISTINCT title FROM chunks WHERE LOWER(title) LIKE '%mobilite%' OR LOWER(title) LIKE '%mobility%'")
print("\nMobility-related documents:")
for r in cur.fetchall():
    print(f"  - {r[0]}")

cur.execute("SELECT title, COUNT(*) as cnt FROM chunks GROUP BY title ORDER BY cnt DESC LIMIT 20")
print("\nTop documents by chunk count:")
for r in cur.fetchall():
    print(f"  {r[1]:4d} - {r[0]}")

cur.close()
conn.close()
