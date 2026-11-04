"""Analyze URL mappings and build file-lookup script"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))
from rag.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
import psycopg2, psycopg2.extras

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cur = conn.cursor()

# Get all distinct URLs
cur.execute("SELECT DISTINCT site, url FROM chunks ORDER BY site, url")
rows = cur.fetchall()

# Show web URL path components
from urllib.parse import urlparse
for s, u in rows:
    if not u:
        continue
    if u.startswith("http"):
        parsed = urlparse(u)
        path = parsed.path.strip("/").lower()
        print(f"URL  | {s:4s} | {path:50s} | {u}")
    elif u.endswith(".pdf") or u.endswith(".PDF"):
        print(f"PDF  | {s:4s} | {u}")
    elif u.endswith(".docx"):
        print(f"DOCX | {s:4s} | {u}")
    else:
        print(f"OTHER| {s:4s} | {u}")
