"""Eval with HyDE — hypothetical answer replaces query for dense retrieval."""
import os, sys, json, argparse
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_hybrid_rag import (
    load_chunks, _doc_key, DenseRetriever, BM25Retriever,
    load_cross_encoder, weighted_rrf, rerank, compute_metrics,
    print_metrics, PROJECT_ROOT
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Eval with HyDE")
    parser.add_argument("--corpus", default="data/synthetic/corpus_merged.json")
    parser.add_argument("--hyde", required=True, help="HyDE queries JSON")
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

    hyde_path = Path(args.hyde)
    if not hyde_path.is_absolute():
        hyde_path = PROJECT_ROOT / args.hyde
    with open(hyde_path, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Corpus: {len(corpus)} | Queries: {len(queries)}")
    print(f"  types: {dict(Counter(q['target_type'] for q in queries))}")

    print("\nLoading models...")
    dense_model = SentenceTransformer("intfloat/multilingual-e5-large")
    corpus_map = {_doc_key(c): c for c in corpus}
    # HyDE: re-embed corpus with "passage:" prefix too (same as DenseRetriever does)
    dense = DenseRetriever(corpus, dense_model)
    sparse = BM25Retriever(corpus)
    ce = load_cross_encoder("BAAI/bge-reranker-v2-m3")

    ranks_dense = []
    ranks_hybrid = []
    ranks_rerank = []

    for row in tqdm(queries, desc="HyDE eval"):
        target = row["target_key"]
        original_q = row["query"]
        hyde_q = row.get("augmented_query", original_q)

        # Dense: use HyDE answer as query
        d_res = dense.search(hyde_q, k=args.top_k_dense)
        d_keys = [doc for doc, _ in d_res]
        ranks_dense.append(d_keys.index(target) + 1 if target in d_keys else None)

        # Hybrid: BM25 on original query, dense on HyDE query
        s_res = sparse.search(original_q, k=args.top_k_sparse)
        h_keys = weighted_rrf(d_res, s_res, k=args.rrf_k,
                              dense_weight=args.dense_weight, sparse_weight=args.sparse_weight)
        ranks_hybrid.append(h_keys.index(target) + 1 if target in h_keys else None)

        # Rerank on original query
        candidates = h_keys[:args.top_k_rerank]
        reranked = rerank(original_q, candidates, corpus_map, ce)
        ranks_rerank.append(reranked.index(target) + 1 if target in reranked else None)

    metrics = {
        "dense":  compute_metrics(ranks_dense),
        "hybrid": compute_metrics(ranks_hybrid),
        "rerank": compute_metrics(ranks_rerank),
    }

    print("\n=== Overall Metrics (HyDE) ===")
    for method in ["dense", "hybrid", "rerank"]:
        print_metrics(method, metrics[method])

if __name__ == "__main__":
    main()
