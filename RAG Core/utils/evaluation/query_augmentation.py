"""
query_augmentation.py — Pre-process eval queries with LLM augmentation.
Modes:
  rewrite  : rewrite query in 3 variants → fuse retrieval
  hyde     : generate hypothetical answer → use as dense query
  stepback : generate broader question → combine with original
"""
import os, sys, json, time, argparse
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_hybrid_rag import (
    load_chunks, _doc_key, build_eval_queries, load_real_chunks, PROJECT_ROOT
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

def _ollama(prompt: str, max_tokens: int = 200) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "options": {"num_predict": max_tokens},
    }, timeout=120)
    return resp.json()["response"].strip()

def rewrite_query(query: str, target_type: str) -> list:
    prompt = f"""Tu es un assistant qui reformule des questions d'étudiants sur les programmes universitaires IIT/ISB en Tunisie. Réécris la question suivante en 3 variantes différentes qui conservent le même sens mais utilisent un vocabulaire différent. Les reformulations doivent être en FRANÇAIS.

Question: {query}

Retourne uniquement les 3 reformulations, une par ligne, sans numérotation."""
    try:
        resp = _ollama(prompt, 200)
        lines = [l.strip("- *0123456789. ") for l in resp.strip().split("\n") if l.strip()]
        # Filter to at least 10 chars
        lines = [l for l in lines if len(l) > 10][:3]
        if lines:
            return [query] + lines  # original + 3 rewrites
        return [query]
    except Exception as e:
        print(f"  [warn] Ollama rewrite failed: {e}")
        return [query]

def hyde_query(query: str, target_type: str) -> str:
    prompt = f"""Tu es un étudiant qui pose une question sur les programmes universitaires IIT/ISB en Tunisie. Génère un paragraphe hypothétique de réponse idéale qui contiendrait toutes les informations nécessaires pour répondre à cette question. Le paragraphe doit être factuel et informatif. En FRANÇAIS.

Question: {query}

Réponse hypothétique:"""
    try:
        response = _ollama(prompt, 300)
        return response
    except Exception as e:
        print(f"  [warn] Ollama HyDE failed: {e}")
        return query

def stepback_query(query: str, target_type: str) -> str:
    domain_map = {
        "mobility": "la mobilité internationale, les échanges, les stages à l'étranger, les bourses, les frais de scolarité",
        "study_plan": "les plans d'études, les programmes académiques, les modules, les cours",
        "general": "les informations générales sur l'université, les admissions, les procédures",
    }
    domain = domain_map.get(target_type, "les programmes universitaires")
    prompt = f"""Tu es un assistant universitaire. La question suivante concerne {domain} à l'IIT/ISB Tunisie. Génère une question plus large et générale qui permettrait de trouver le contexte pertinent.

Question spécifique: {query}

Question plus large:"""
    try:
        response = _ollama(prompt, 150)
        return response
    except Exception as e:
        print(f"  [warn] Ollama stepback failed: {e}")
        return query


def main():
    parser = argparse.ArgumentParser(description="Augment eval queries via Ollama")
    parser.add_argument("mode", choices=["rewrite", "hyde", "stepback"])
    parser.add_argument("--corpus", default="data/synthetic/corpus_merged.json")
    parser.add_argument("--real", default="both")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    corpus = load_chunks(str(PROJECT_ROOT / args.corpus))
    real_chunks = load_real_chunks(args.real)
    queries = build_eval_queries(real_chunks, args.queries, stratify=True, seed=args.seed)
    print(f"Loaded {len(queries)} queries")

    fn_map = {"rewrite": rewrite_query, "hyde": hyde_query, "stepback": stepback_query}
    fn = fn_map[args.mode]

    augmented = []
    t0 = time.time()
    for i, q in enumerate(queries):
        result = fn(q["query"], q["target_type"])
        if args.mode == "rewrite":
            augmented.append({**q, "query_variants": result})
        else:
            augmented.append({**q, "augmented_query": result})
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(queries) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(queries)}] {q['query'][:60]}... -> done ({remaining:.0f}s remaining)" if args.mode == "rewrite" else f"  [{i+1}/{len(queries)}] done")
        sys.stdout.flush()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(augmented)} augmented queries to {output_path}")

if __name__ == "__main__":
    main()
