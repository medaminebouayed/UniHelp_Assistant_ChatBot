"""Replicate API logic exactly to find where fee doc gets filtered."""
import sys, os, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))

from rag.pipeline import query as rag_query
from rag.graph_index import _extract_entities
from rag.retriever import _accent_fold
from rag.store import get_conn

q = "C'est quoi les frais de scolarite a IIT ?"

results = rag_query(q, top_k=30, use_graph=False)
print(f"Pipeline returned {len(results)} results")

# Check which have "frais" in title
fee_results = [r for r in results if "frais" in (r.get("title","") or "").lower()]
print(f"Fee docs in results: {len(fee_results)}")
for r in fee_results:
    print(f"  title={r.get('title','')}, site={r.get('site','')}")

# Replicate API scoring
query_kw = _extract_entities(q)
print(f"\nQuery keywords: {query_kw}")

total = len(results)
scored = []
for i, r in enumerate(results):
    title = _accent_fold(r.get("title", "") or "").lower()
    section = _accent_fold(r.get("section_title", "") or "").lower()
    kw_matches = sum(1 for kw in query_kw if kw.lower() in title or kw.lower() in section)
    kw_score = kw_matches / max(len(query_kw), 1) if query_kw else 0
    pos_score = 1.0 - (i / max(total, 1))
    combined = 0.4 * kw_score + 0.6 * pos_score
    scored.append((combined, r))

scored.sort(key=lambda x: -x[0])

print(f"\nScored top 15:")
for s, r in scored[:15]:
    title = r.get("title", "")
    kw = sum(1 for kw in query_kw if kw.lower() in title.lower())
    fee = " <-- FEE" if "frais" in title.lower() else ""
    print(f"  score={s:.3f}, kw={kw}, {title[:70]}{fee}")

# Threshold filter
filtered = [r for s, r in scored if s > 0.2]
print(f"\nAfter threshold filter: {len(filtered)}")

# Check if fee doc survives
fee_after = [r for r in filtered if "frais" in (r.get("title","") or "").lower()]
print(f"Fee docs after threshold: {len(fee_after)}")

# Dedup
MAX_PER_DOC = 3
doc_counts = {}
final = []
for r in filtered:
    title = r.get("title", "") or "unknown"
    cnt = doc_counts.get(title, 0)
    if cnt >= MAX_PER_DOC:
        continue
    doc_counts[title] = cnt + 1
    final.append(r)

fee_after_dedup = [r for r in final if "frais" in (r.get("title","") or "").lower()]
print(f"Fee docs after dedup: {len(fee_after_dedup)}")

# 5-gram filtering
_BAN5 = frozenset(["ation", "ement", "ment ", "tion ", "satio"])
def _ng5(text):
    s = _accent_fold(text).lower()
    if len(s) < 5:
        return set()
    return {s[i:i+5] for i in range(len(s)-4)} - _BAN5
kw5 = set()
for k in query_kw:
    kw5 |= _ng5(k)
print(f"kw5: {kw5}")

def _overlap(r):
    t = _ng5(r.get("title", "") or "")
    s = _ng5(r.get("section_title", "") or "")
    return len(kw5 & (t | s))

max_ov = max((_overlap(r) for r in final), default=0)
print(f"max_ov: {max_ov}")

for r in final:
    ov = _overlap(r)
    title = r.get("title","")
    if "frais" in title.lower():
        print(f"Fee doc '{title[:60]}' overlap: {ov}")

if max_ov >= 3:
    threshold = max(1, max_ov - 2)
    print(f"5-gram filter threshold: {threshold}, keeping {sum(1 for r in final if _overlap(r) >= threshold)} docs")
    final = [r for r in final if _overlap(r) >= threshold]

fee_after_ng5 = [r for r in final if "frais" in (r.get("title","") or "").lower()]
print(f"Fee docs after 5-gram: {len(fee_after_ng5)}")

# Site filter
r = re.search(r'\biit\b', q.lower())
print(f"\nSite in query: {'iit' if r else 'none'}")

# Print final titles
print(f"\nFinal {len(final)} titles:")
for r in final:
    title = r.get("title","")
    fee = " <-- FEE" if "frais" in title.lower() else ""
    print(f"  {title[:70]}{fee}")
