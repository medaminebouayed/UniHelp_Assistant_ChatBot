"""Debug with API's actual top_k=30."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.retriever import hybrid_search, _expand_query, _accent_fold
from rag.reranker import rerank
from rag.store import get_conn
from rag.graph_index import _extract_entities

q = "C'est quoi les frais de scolarite a IIT ?"

hybrid = hybrid_search(q, top_k_dense=80, top_k_sparse=100, use_graph=False)
reranked = rerank(q, hybrid[:30])
print(f"Reranked (top 30): {len(reranked)}")

conn = get_conn()
cur = conn.cursor()

# Check if fee doc is in reranked
for i, dk in enumerate(reranked):
    cur.execute("SELECT title FROM chunks WHERE doc_key = %s", (dk,))
    row = cur.fetchone()
    title = row[0] if row else "N/A"
    if "frais" in title.lower():
        print(f"  Fee doc at reranked[{i}]: {title}")

# Now replicate the API's scoring/filtering
query_kw = _extract_entities(q)
print(f"\nQuery keywords: {query_kw}")

results = []
for dk in reranked:
    cur.execute("""
        SELECT doc_key, site, title, section_title, content_type,
               page_content, enriched_content, questions, keywords, url
        FROM chunks WHERE doc_key = %s
    """, (dk,))
    row = cur.fetchone()
    if row:
        results.append({
            "doc_key": row[0], "site": row[1], "title": row[2],
            "section_title": row[3], "content_type": row[4],
            "page_content": row[5], "enriched_content": row[6],
            "questions": row[7], "keywords": row[8], "url": row[9],
        })

total = len(results)
scored = []
for i, r in enumerate(results):
    title = _accent_fold(r["title"] or "").lower()
    section = _accent_fold(r["section_title"] or "").lower()
    kw_matches = sum(1 for kw in query_kw if kw.lower() in title or kw.lower() in section)
    kw_score = kw_matches / max(len(query_kw), 1) if query_kw else 0
    pos_score = 1.0 - (i / max(total, 1))
    combined = 0.4 * kw_score + 0.6 * pos_score
    scored.append((combined, r))

scored.sort(key=lambda x: -x[0])

# Threshold filter
filtered = [r for s, r in scored if s > 0.2]
print(f"\nAfter threshold filter: {len(filtered)} (fee: {sum(1 for r in filtered if 'frais' in (r['title'] or '').lower())})")

# If < 3, keep top 10
if len(filtered) < 3:
    filtered = [r for _, r in scored[:10]]

# Dedup
MAX_PER_DOC = 3
doc_counts = {}
final = []
for r in filtered:
    title = r["title"] or "unknown"
    cnt = doc_counts.get(title, 0)
    if cnt >= MAX_PER_DOC:
        continue
    doc_counts[title] = cnt + 1
    final.append(r)

print(f"After dedup: {len(final)} (fee: {sum(1 for r in final if 'frais' in (r['title'] or '').lower())})")

# If < 3, fallback
if len(final) < 3:
    final = results[:30]

# 5-gram filter
_BAN5 = frozenset(["ation", "ement", "ment ", "tion ", "satio"])
def _ng5(text):
    s = _accent_fold(text).lower()
    if len(s) < 5:
        return set()
    return {s[i:i+5] for i in range(len(s)-4)} - _BAN5
kw5 = set()
for k in query_kw:
    kw5 |= _ng5(k)

def _overlap(r):
    t = _ng5(r.get("title", "") or "")
    s = _ng5(r.get("section_title", "") or "")
    return len(kw5 & (t | s))

max_ov = max((_overlap(r) for r in final), default=0)
if max_ov >= 3:
    threshold = max(1, max_ov - 2)
    print(f"5-gram filter: max_ov={max_ov}, threshold={threshold}")
    print(f"  Fee doc overlap: {[_overlap(r) for r in final if 'frais' in (r['title'] or '').lower()]}")
    final = [r for r in final if _overlap(r) >= threshold]

print(f"Final: {len(final)} (fee: {sum(1 for r in final if 'frais' in (r['title'] or '').lower())})")
for r in final:
    title = r["title"]
    fee = " <-- FEE" if "frais" in (title or "").lower() else ""
    print(f"  {title[:60]}{fee}")

cur.close()
conn.close()
