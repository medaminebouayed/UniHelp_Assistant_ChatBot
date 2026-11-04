"""
RAG Performance & Accuracy Test Suite

Tests the RAG API for:
  - Query accuracy (expected sources, correct answers)
  - Acronym handling (IIT/ISB must NOT be expanded)
  - Source diversity (MAX_PER_DOC dedup)
  - Site filtering (queries mentioning IIT/ISB)
  - Typo tolerance
  - Edge cases (empty, short, irrelevant)
  - Latency

Usage:
    python tests/test_rag_performance.py              # localhost:8002
    python tests/test_rag_performance.py --url http://rag-api:8002
    python tests/test_rag_performance.py --url http://localhost:8000 --via-backend
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from typing import Any

PASS = 0
FAIL = 0
SKIP = 0

SEP = "=" * 70
SUB = "-" * 70


def post_json(url: str, data: dict) -> dict | None:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [NETWORK ERROR] {e}")
        return None


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def skip(name: str, reason: str = ""):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name} -- {reason}")


# ────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ────────────────────────────────────────────────────────────────────────────

QUERIES: list[dict[str, Any]] = [
    {
        "q": "Quelles sont les accreditations de l'IIT ?",
        "expect_title": "Accreditations",
        "check": lambda r, s: "EUR-ACE" in (r or "").upper() or "EURO-INF" in (r or "").upper(),
        "desc": "EUR-ACE / EURO-INF doivent etre cites",
    },
    {
        "q": "C'est quoi les frais de scolarite a IIT ?",
        "expect_title": "frais",
        "check": lambda r, s: any("frais" in (t.get("title", "") or "").lower() for t in s),
        "desc": "Sources sur les frais de scolarite IIT",
    },
    {
        "q": "Quels sont les programmes de mobilite a ISB ?",
        "expect_title": "international",
        "check": lambda r, s: any("mobility" in (t.get("type", "") or "").lower() for t in s),
        "desc": "Programmes de mobilite ISB",
    },
    {
        "q": "Qu'est-ce que l'Institut International de Technologie propose comme formations ?",
        "expect_title": "IIT",
        "check": lambda r, s: any("IIT" in (t.get("site", "") or "") for t in s),
        "desc": "Formations IIT (nom complet en question)",
    },
    {
        "q": "Raconte-moi une blague",
        "check": lambda r, s: len(s) == 0,
        "desc": "Question hors-sujet -> pas de sources",
    },
]


ACRONYM_QUERIES = [
    "Quels sont les frais a IIT ?",
    "Parle-moi de ISB",
    "Comment postuler a IIT ?",
]

TYPO_QUERIES = [
    ("accreditations IIT", "accreditations"),
    ("frais scolarite", "frais"),
    ("programme mobilite ISB", "mobilite"),
]


def test_query(entry: dict, rag_url: str, via_backend: bool) -> dict | None:
    q = entry["q"]
    if via_backend:
        url = f"{rag_url}/chat"
        body = {"conversation_id": 0, "question": q, "use_graph": True, "top_k": 30}
    else:
        url = f"{rag_url}/query"
        body = {"question": q, "top_k": 30, "use_graph": True}

    start = time.perf_counter()
    result = post_json(url, body)
    elapsed = time.perf_counter() - start

    if result is None:
        return None

    answer = ""
    sources: list = []
    if via_backend:
        answer = result.get("reponse", "")
        sources = result.get("sources", [])
    else:
        answer = result.get("answer", "")
        sources = result.get("sources", [])

    return {"answer": answer, "sources": sources, "latency": round(elapsed, 3)}


def run_suite(rag_url: str, via_backend: bool):
    global PASS, FAIL, SKIP
    PASS = FAIL = SKIP = 0

    print(f"\n{SEP}")
    print(f"  RAG PERFORMANCE TEST SUITE")
    print(f"  Target: {rag_url}  (via_backend={via_backend})")
    print(f"{SEP}")

    # ── 1. Accuracy & Expected Sources ────────────────────────────────
    print(f"\n{SUB}")
    print("  1. ACCURACY & EXPECTED SOURCES")
    print(f"{SUB}")

    for entry in QUERIES:
        q = entry["q"]
        desc = entry["desc"]
        print(f"\n  Q: {q}")
        result = test_query(entry, rag_url, via_backend)
        if result is None:
            FAIL += 1
            print(f"  [FAIL] {desc} -- No response (network error)")
            continue

        answer = result["answer"]
        sources = result["sources"]
        print(f"  Latency: {result['latency']}s  |  Sources: {len(sources)}")

        if entry.get("check"):
            ok = entry["check"](answer, sources)
            check(desc, ok, "Custom check failed")

        if entry.get("expect_title"):
            import unicodedata, re as _re
            def fold(s):
                return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
            kw = entry["expect_title"]
            kw_folded = fold(kw)
            check_label = f"Source avec titre contenant '{kw}'"
            ok = False
            for s in sources:
                src = (s.get("title", "") or "") + " " + (s.get("section", "") or "")
                src_folded = fold(src)
                if kw_folded in src_folded:
                    ok = True
                    break
                words = _re.findall(r"[a-z0-9]+", src_folded)
                if any(kw_folded.startswith(w) or w.startswith(kw_folded) for w in words):
                    ok = True
                    break
            check(check_label, ok,
                  f"Titres: {[s.get('title','')[:60] for s in sources[:5]]}")

    # ── 2. Acronym Handling ─────────────────────────────────────────
    print(f"\n{SUB}")
    print("  2. ACRONYM HANDLING (IIT/ISB must NOT be expanded)")
    print(f"{SUB}")

    import re
    for q in ACRONYM_QUERIES:
        print(f"\n  Q: {q}")
        result = test_query({"q": q}, rag_url, via_backend)
        if result is None:
            FAIL += 1
            print(f"  [FAIL] No response")
            continue
        answer = result["answer"]
        bad = re.findall(r'IIT\s*\([^)]{3,}\)', answer, re.IGNORECASE)
        bad += re.findall(r'ISB\s*\([^)]{3,}\)', answer, re.IGNORECASE)
        ok = len(bad) == 0
        check(f"Pas d'expansion de sigle: {q[:50]}", ok,
              f"Trouve: {bad[:3]}" if bad else "")
        print(f"  Answer (first 200): {answer[:200]}")

    # ── 3. Source Diversity (MAX_PER_DOC) ───────────────────────────
    print(f"\n{SUB}")
    print("  3. SOURCE DIVERSITY (<=3 chunks per doc)")
    print(f"{SUB}")

    q = "Que propose l'IIT comme formations et services ?"
    print(f"\n  Q: {q}")
    result = test_query({"q": q}, rag_url, via_backend)
    if result:
        sources = result["sources"]
        from collections import Counter
        doc_counts = Counter(s.get("title", "") for s in sources)
        exceeds = {t: c for t, c in doc_counts.items() if c > 3}
        ok = len(exceeds) == 0
        check("MAX_PER_DOC respecte (<=3 par titre)", ok,
              f"Depassements: {exceeds}" if exceeds else "")
        print(f"  Distribution: {dict(doc_counts.most_common(8))}")
        print(f"  Latency: {result['latency']}s")

    # ── 4. Site Filtering ───────────────────────────────────────────
    print(f"\n{SUB}")
    print("  4. SITE FILTERING (query mentions IIT/ISB)")
    print(f"{SUB}")

    q = "Comment se deroule le processus d'inscription a ISB ?"
    print(f"\n  Q: {q}")
    result = test_query({"q": q}, rag_url, via_backend)
    if result:
        sources = result["sources"]
        sites = set(s.get("site", "").upper() for s in sources if s.get("site"))
        iit_sources = [s for s in sources if s.get("site", "").upper() == "IIT"]
        ok = len(iit_sources) == 0 or len(sites) == 1
        check("Filtrage IIT/ISB respecte", ok,
              f"Sites dans les sources: {sites} (IIT count: {len(iit_sources)})")
        print(f"  Sites: {sites}, Sources total: {len(sources)}")

    # ── 5. Typo Tolerance ───────────────────────────────────────────
    print(f"\n{SUB}")
    print("  5. TYPO TOLERANCE")
    print(f"{SUB}")

    for q, keyword in TYPO_QUERIES:
        print(f"\n  Q: {q}")
        result = test_query({"q": q}, rag_url, via_backend)
        if result is None:
            FAIL += 1
            print(f"  [FAIL] No response")
            continue
        answer = result["answer"]
        sources = result["sources"]
        ok = len(sources) > 0
        check(f"Reponse avec sources: {q[:50]}", ok, "0 sources")
        if sources:
            import re, unicodedata
            def fold(s):
                return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
            kw_ok = False
            kw_folded = fold(keyword)
            for s in sources:
                src = (s.get("title", "") or "") + " " + (s.get("section", "") or "")
                src_folded = fold(src)
                if kw_folded in src_folded:
                    kw_ok = True
                    break
                words = re.findall(r"[a-z0-9]+", src_folded)
                if any(kw_folded.startswith(w) or w.startswith(kw_folded) for w in words):
                    kw_ok = True
                    break
            check(f"Source pertinente (contient '{keyword}')", kw_ok,
                  f"Titres: {[s.get('title','')[:40] for s in sources[:5]]}")
        print(f"  Sources: {len(sources)}, Latency: {result['latency']}s")

    # ── 6. Edge Cases ───────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  6. EDGE CASES")
    print(f"{SUB}")

    edge_queries = [
        ("", "Requete vide"),
        ("a", "Requete tres courte (1 char)"),
        ("IIT", "Requete tres courte (3 chars)"),
        ("xyzzy_nonexistent_12345", "Requete sans document correspondant"),
    ]

    for q, desc in edge_queries:
        print(f"\n  Q: '{q}'")
        result = test_query({"q": q}, rag_url, via_backend)
        if result is None:
            FAIL += 1
            print(f"  [FAIL] {desc} -- No response")
            continue
        ok = len(result.get("answer", "")) > 0
        check(desc, ok, "Reponse vide ou erreur")
        print(f"  Answer (first 150): {result['answer'][:150]}")
        print(f"  Latency: {result['latency']}s")

    # ── 7. Latency ──────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  7. LATENCY SUMMARY")
    print(f"{SUB}")

    latencies = []
    for entry in QUERIES:
        result = test_query(entry, rag_url, via_backend)
        if result:
            latencies.append(result["latency"])
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  Avg latency: {avg:.2f}s  |  Min: {min(latencies):.2f}s  |  Max: {max(latencies):.2f}s")
        check("Latence moyenne < 20s", avg < 20.0, f"Moyenne: {avg:.2f}s")

    # ── Summary ─────────────────────────────────────────────────────
    total = PASS + FAIL + SKIP
    print(f"\n{SEP}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped (of {total} checks)")
    print(f"{SEP}\n")

    return FAIL == 0


def main():
    parser = argparse.ArgumentParser(description="RAG Performance & Accuracy Test Suite")
    parser.add_argument("--url", default="http://127.0.0.1:8002", help="RAG API or Backend URL")
    parser.add_argument("--via-backend", action="store_true", help="Route through backend /chat endpoint")
    args = parser.parse_args()

    success = run_suite(args.url, args.via_backend)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
