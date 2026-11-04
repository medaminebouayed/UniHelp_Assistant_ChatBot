"""Check plan d'etude chunks in DB"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))
from rag.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
import psycopg2
conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cur = conn.cursor()

cur.execute("SELECT DISTINCT title FROM chunks WHERE title ILIKE '%plan%' OR title ILIKE '%etude%' ORDER BY title")
print("Plans d'etude:")
for r in cur.fetchall():
    print(f"  {r[0]}")

cur.execute("SELECT DISTINCT title FROM chunks WHERE title ILIKE '%informatique%' OR title ILIKE '%logiciel%' ORDER BY title")
print("\nGenie informatique/logiciel:")
for r in cur.fetchall():
    print(f"  {r[0]}")

cur.execute("SELECT doc_key, section_title FROM chunks WHERE doc_key ILIKE '%plan_etudelicence%' LIMIT 5")
print("\nLicence Info plan doc_keys:")
for r in cur.fetchall():
    print(f"  {r[0]} | {r[1]}")
