"""Check special URL cases in chunks table"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))
from rag.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
import psycopg2

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cur = conn.cursor()
cur.execute("SELECT url, COUNT(*) FROM chunks WHERE url NOT LIKE 'http%%' AND url NOT LIKE '%%.pdf' AND url NOT LIKE '%%.PDF' AND url NOT LIKE '%%.docx' GROUP BY url")
for r in cur.fetchall():
    print(repr(r[0]), '->', r[1])
