"""
build_bm25_index.py — Build BM25 keyword index for hybrid retrieval
====================================================================
Indexes bm25_text field (NOT page_content) to avoid breadcrumb pollution.

French-aware tokenizer handles accents and module codes like "GC1 101".

Dependencies:
    pip install rank-bm25

Usage:
    py utils/build_bm25_index.py --corpus data/synthetic/corpus_final.json
    py utils/build_bm25_index.py --corpus data/synthetic/corpus_final.json --output bm25_index.pkl
"""

import re
import json
import pickle
import argparse
from pathlib import Path

from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[error] rank_bm25 not installed. Install: pip install rank-bm25")
    raise SystemExit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_corpus(corpus_path: str) -> list:
    path = Path(corpus_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[error] Corpus not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[info] Loaded {len(data)} chunks from {path}")
    return data


def french_tokenize(text: str) -> list:
    """French-aware tokenizer.

    - Keeps French accented characters: é, à, è, ù, â, ê, î, ô, û, ç, œ, æ
    - Preserves module codes: "GC1", "GM2-5", "BIM"
    - Preserves hyphenated terms (compound words common in French academic text)
    - Drops single-char tokens ("d", "l", "s" from French elisions)
    """
    tokens = re.findall(r"[a-zéàèùâêîôûçœæ0-9][a-zéàèùâêîôûçœæ0-9\-]*", text.lower())
    return [t for t in tokens if len(t) >= 2]


def build_index(corpus_path: str, output_path: str, field: str = "bm25_text"):
    """Build BM25Okapi index from the specified field."""
    chunks = load_corpus(corpus_path)
    if not chunks:
        return

    print(f"[bm25] Indexing field: '{field}'")
    print(f"[bm25] Tokenizing {len(chunks)} documents...")

    tokenized_docs = []
    doc_keys = []
    stats = {"empty": 0, "total": 0}

    for chunk in tqdm(chunks):
        text = chunk.get(field, chunk.get("page_content", ""))
        if not text or not text.strip():
            stats["empty"] += 1
            tokenized_docs.append([])
        else:
            tokenized_docs.append(french_tokenize(text))

        meta = chunk.get("metadata", {})
        doc_key = meta.get("doc_key") or chunk.get("doc_key") or \
                  f"{meta.get('site', 'unknown')}_{meta.get('title', 'untitled')}_{hash(chunk.get('page_content', '')) % 10**6}"
        doc_keys.append(doc_key)
        stats["total"] += 1

    print(f"[bm25] Building BM25Okapi index ({stats['total']} docs, {stats['empty']} empty)...")
    bm25 = BM25Okapi(tokenized_docs)

    index_data = {
        "bm25": bm25,
        "doc_keys": doc_keys,
        "tokenized_docs": tokenized_docs,
        "corpus_path": str(corpus_path),
        "field": field,
    }

    output = Path(output_path)
    with open(output, "wb") as f:
        pickle.dump(index_data, f)

    print(f"[bm25] Index saved to {output.resolve()}")
    print(f"[bm25] {len(doc_keys)} documents indexed")

    # Quick sanity check
    sample_query = "master marketing digital"
    tokens = french_tokenize(sample_query)
    scores = bm25.get_scores(tokens)
    top_n = sorted(zip(doc_keys, scores), key=lambda x: -x[1])[:3]
    print(f"[bm25] Sanity check — top-3 for '{sample_query}':")
    for key, score in top_n:
        if score > 0:
            print(f"        {key}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 keyword index")
    parser.add_argument("--corpus", default="data/synthetic/corpus_final.json",
                        help="Path to merged corpus JSON")
    parser.add_argument("--output", default="data/bm25_index.pkl",
                        help="Output pickle path")
    parser.add_argument("--field", default="bm25_text",
                        help="Field to index (default: bm25_text)")
    args = parser.parse_args()

    build_index(args.corpus, args.output, field=args.field)
