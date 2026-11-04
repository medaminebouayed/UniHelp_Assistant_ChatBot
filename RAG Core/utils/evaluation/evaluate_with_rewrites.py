"""
evaluate_with_rewrites.py — Eval using multi-variant query rewriting.
For each query, all 4 variants (original + 3 rewrites) are searched independently,
then fused via RRF before passing to the reranker.
"""
import os, sys, json, argparse
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_hybrid_rag import (
    load_chunks, _doc_key, DenseRetriever, BM25Retriever,
    GraphRetriever, load_cross_encoder, weighted_rrf, rerank,
    compute_metrics, print_metrics, PROJECT_ROOT
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Eval with query rewrites")
    parser.add_argument("--corpus", default="data/synthetic/corpus_merged.json")
    parser.add_argument("--rewrites", required=True, help="Path to rewritten queries JSON")
    parser.add_argument("--dense-weight", type=float, default=0.55)
    parser.add_argument("--sparse-weight", type=float, default=0.45)
    parser.add_argument("--rrf-k", type=int, default=40)
    parser.add_argument("--top-k-dense", type=int, default=150)
    parser.add_argument("--top-k-sparse", type=int, default=200)
    parser.add_argument("--top-k-rerank", type=int, default=50)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / args.corpus
    corpus = load_chunks(str(corpus_path))

    rewrites_path = Path(args.rewrites)
    if not rewrites_path.is_absolute():
        rewrites_path = PROJECT_ROOT / args.rewrites
    with open(rewrites_path, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Corpus: {len(corpus)} chunks | Queries: {len(queries)}")
    print(f"  query types: {dict(Counter(q['target_type'] for q in queries))}")

    print("\nLoading models...")
    dense_model = SentenceTransformer("intfloat/multilingual-e5-large")
    dense = DenseRetriever(corpus, dense_model)
    sparse = BM25Retriever(corpus)
    ce = load_cross_encoder("BAAI/bge-reranker-v2-m3")

    corpus_map = {_doc_key(c): c for c in corpus}

    ranks_dense  = []
    ranks_hybrid = []
    ranks_rerank = []

    for row in tqdm(queries, desc="Evaluating with rewrites"):
        target = row["target_key"]
        variants = row.get("query_variants", [row["query"]])
        # Unique variants (deduplicate)
        seen = set()
        unique_variants = []
        for v in variants:
            vn = v.strip().lower()
            if vn not in seen:
                seen.add(vn)
                unique_variants.append(v)

        # ── Dense: run ALL variants, fuse via RRF ──────────────────
        all_dense_results = []
        for vq in unique_variants:
            d_res = dense.search(vq, k=args.top_k_dense // len(unique_variants))
            all_dense_results.extend(d_res)
        # Deduplicate best score per doc_key
        best_score = {}
        for dk, sc in all_dense_results:
            best_score[dk] = max(best_score.get(dk, sc), sc)
        d_keys = sorted(best_score, key=best_score.get, reverse=True)[:args.top_k_dense]
        ranks_dense.append(d_keys.index(target) + 1 if target in d_keys else None)

        # ── Hybrid: RRF over all variant searches ──────────────────
        all_hybrid_pools = []
        for vq in unique_variants:
            d_res = dense.search(vq, k=args.top_k_dense)
            s_res = sparse.search(vq, k=args.top_k_sparse)
            h_keys = weighted_rrf(d_res, s_res, k=args.rrf_k,
                                  dense_weight=args.dense_weight, sparse_weight=args.sparse_weight)
            all_hybrid_pools.extend(h_keys)
        # Deduplicate + re-rank by frequency (fusion)
        freq = Counter(all_hybrid_pools)
        h_keys_fused = [dk for dk, _ in freq.most_common()]
        ranks_hybrid.append(h_keys_fused.index(target) + 1 if target in h_keys_fused else None)

        # ── Rerank on fused hybrid pool ────────────────────────────
        candidates = h_keys_fused[:args.top_k_rerank]
        reranked = rerank(row["query"], candidates, corpus_map, ce)
        ranks_rerank.append(reranked.index(target) + 1 if target in reranked else None)

    metrics = {
        "dense":  compute_metrics(ranks_dense),
        "hybrid": compute_metrics(ranks_hybrid),
        "rerank": compute_metrics(ranks_rerank),
    }

    print("\n=== Overall Metrics (with Query Rewrites) ===")
    for method in ["dense", "hybrid", "rerank"]:
        print_metrics(method, metrics[method])

    print("\n=== Delta vs Baseline Rerank ===")
    for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg"]:
        print(f"  {k}: {metrics['rerank'][k]:.4f}")

    return metrics

if __name__ == "__main__":
    main()
