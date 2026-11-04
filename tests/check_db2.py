"""Deep DB diagnostics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.store import get_conn

conn = get_conn()
cur = conn.cursor()

# Check for ANY chunks mentioning accreditation
cur.execute("SELECT title, COUNT(*) FROM chunks WHERE LOWER(title) LIKE '%%accred%%' OR LOWER(title) LIKE '%%accredit%%' GROUP BY title")
rows = cur.fetchall()
print(f"Accreditation mentions: {len(rows)}")
for r in rows:
    print(f"  title='{r[0]}' count={r[1]}")

# Check if there are chunks with enriched_content containing EUR-ACE
cur.execute("SELECT doc_key, site, title FROM chunks WHERE page_content LIKE '%%EUR-ACE%%' LIMIT 5")
rows = cur.fetchall()
print(f"EUR-ACE chunks: {len(rows)}")
for r in rows:
    print(f"  key={str(r[0])[:60]}, site={r[1]}, title={r[2]}")

# Check chunks without embeddings
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL AND indexed_at IS NOT NULL")
print(f"Indexed but no embedding: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
print(f"With embedding: {cur.fetchone()[0]}")

# Check FORMATION - IIT chunks for embedding
cur.execute("SELECT COUNT(*) FROM chunks WHERE LOWER(title) = 'formation - iit' AND embedding IS NOT NULL")
print(f"FORMATION - IIT with embedding: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM chunks WHERE LOWER(title) = 'formation - iit' AND embedding IS NULL")
print(f"FORMATION - IIT without embedding: {cur.fetchone()[0]}")

# Sample some chunks without embeddings
cur.execute("SELECT doc_key, site, title FROM chunks WHERE embedding IS NULL LIMIT 5")
for r in cur.fetchall():
    print(f"  NO EMBED: key={str(r[0])[:50]}, site={r[1]}, title={r[2]}")

cur.close()
conn.close()
