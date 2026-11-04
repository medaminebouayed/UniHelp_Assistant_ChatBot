"""
grid_search.py — RRF parameter grid search (no-rerank, single model load).
"""
import sys, os, json, time
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_hybrid_rag import (
    load_chunks, _doc_key, DenseRetriever, BM25Retriever,
    weighted_rrf, evaluate, compute_metrics, build_eval_queries,
    load_real_chunks, PROJECT_ROOT,
)
from sentence_transformers import SentenceTransformer

corpus = load_chunks(str(PROJECT_ROOT / "data/synthetic/corpus_merged.json"))
real_chunks = load_real_chunks("both")
queries = build_eval_queries(real_chunks, 300, stratify=True, seed=42)

print(f"Corpus: {len(corpus)} chunks | Queries: {len(queries)}")
print(f"  query types: {dict(Counter(q['target_type'] for q in queries))}")

print("\nLoading E5-large + embedding corpus...")
model = SentenceTransformer("intfloat/multilingual-e5-large")
dense = DenseRetriever(corpus, model)
sparse = BM25Retriever(corpus)

# Grid: weight combinations
weight_grid = [
    (0.50, 0.50),
    (0.55, 0.45),
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
    (0.75, 0.25),
    (0.80, 0.20),
]
k_values = [40, 60, 80, 100]

print(f"\n{'dense_w':>8} {'sparse_w':>9} {'rrf_k':>6} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'hit@10':>8} {'mrr':>7} {'ndcg':>7}")
print("-" * 70)

results = []
for dw, sw in weight_grid:
    for rk in k_values:
        ranks_dense = []
        ranks_hybrid = []
        for row in queries:
            q = row["query"]
            target = row["target_key"]
            d_res = dense.search(q, k=150)
            d_keys = [doc for doc, _ in d_res]
            try:
                pos = d_keys.index(target) + 1
            except ValueError:
                pos = None
            ranks_dense.append(pos)
            s_res = sparse.search(q, k=200)
            h_keys = weighted_rrf(d_res, s_res, k=rk, dense_weight=dw, sparse_weight=sw)
            try:
                pos = h_keys.index(target) + 1
            except ValueError:
                pos = None
            ranks_hybrid.append(pos)

        dense_m = compute_metrics(ranks_dense)
        hybrid_m = compute_metrics(ranks_hybrid)
        h1 = hybrid_m["hit@1"]
        results.append((h1, dw, sw, rk, dense_m, hybrid_m))
        print(f"{dw:>8.2f} {sw:>9.2f} {rk:>6d} {h1:>7.4f} {hybrid_m['hit@3']:>7.4f} {hybrid_m['hit@5']:>7.4f} {hybrid_m['hit@10']:>8.4f} {hybrid_m['mrr']:>7.4f} {hybrid_m['ndcg']:>7.4f}")

# Best result
best = max(results, key=lambda r: r[0])
print(f"\n=== Best: dense_weight={best[1]:.2f}, sparse_weight={best[2]:.2f}, rrf_k={best[3]} ===")
print(f"  Dense:  hit@1={best[4]['hit@1']:.4f}  hit@10={best[4]['hit@10']:.4f}  mrr={best[4]['mrr']:.4f}")
print(f"  Hybrid: hit@1={best[5]['hit@1']:.4f}  hit@10={best[5]['hit@10']:.4f}  mrr={best[5]['mrr']:.4f}")
