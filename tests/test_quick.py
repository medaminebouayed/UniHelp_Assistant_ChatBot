"""Quick RAG query test."""
import json, time, urllib.request, urllib.error

def query(q, url="http://127.0.0.1:8002/query"):
    body = json.dumps({"question": q, "top_k": 30, "use_graph": True}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start
    return data, elapsed

tests = [
    ("Quelles sont les accreditations de l IIT ?", "EUR-ACE"),
    ("C est quoi les frais de scolarite a IIT ?", "frais"),
    ("Raconte-moi une blague", None),
    ("accreditations IIT (faute frappe)", "accreditations"),
    ("Parle-moi de ISB", None),
]

for q, keyword in tests:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    try:
        data, elapsed = query(q)
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        print(f"Time: {elapsed:.2f}s | Sources: {len(sources)}")

        from collections import Counter
        titles = Counter(s.get("title", "") for s in sources)
        for t, c in titles.most_common(5):
            url = next((s.get("url","")[:60] for s in sources if s.get("title")==t), "")
            print(f"  - {t} (x{c}) url={url}")

        if keyword == "EUR-ACE":
            print(f"EUR-ACE in answer: {'EUR-ACE' in answer.upper()}")
            print(f"EURO-INF in answer: {'EURO-INF' in answer.upper()}")
        elif keyword == "frais":
            print(f"frais in any title: {any('frais' in (s.get('title','') or '').lower() for s in sources)}")

        import re
        bad = re.findall(r'IIT\s*\([^)]{3,}\)', answer) + re.findall(r'ISB\s*\([^)]{3,}\)', answer)
        if bad:
            print(f"!! ACRONYM EXPANSION FOUND: {bad}")

        print(f"Answer (first 300): {answer[:300]}")
    except Exception as e:
        print(f"ERROR: {e}")
