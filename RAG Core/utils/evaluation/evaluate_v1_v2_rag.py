"""
A/B retrieval evaluation + intrinsic synthetic quality (dense-only, no HyDE / no graph).

Defaults load IIT+ISB augmented chunks for stratified queries so per-type segments are meaningful.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_HF_TOKEN = os.getenv("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_FILES = [
    PROJECT_ROOT / "data/augmented/iit_chunks_augmented.json",
    PROJECT_ROOT / "data/augmented/isb_chunks_augmented.json",
]


def _norm_header(meta: dict) -> str:
    return (meta.get("section_title") or meta.get("header") or "").strip().lower()


def _doc_key(chunk: dict) -> str:
    m = chunk.get("metadata", {})
    return f"{m.get('site','').strip().lower()}|{m.get('title','').strip().lower()}|{_norm_header(m)}"


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("chunks", [])


def load_real_chunks(real_arg: str) -> list[dict]:
    """Comma-separated paths, keyword 'both', or a single path."""
    paths: list[Path] = []
    if not real_arg or real_arg.strip().lower() in ("both", "all", "default"):
        paths = [p for p in DEFAULT_REAL_FILES if p.exists()]
    else:
        for part in real_arg.split(","):
            p = Path(part.strip())
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No real chunk files found (arg={real_arg!r})")
    out: list[dict] = []
    for p in paths:
        out.extend(load_chunks(p))
    return out


def _normalize_content_type(raw: str) -> str:
    if raw in ("study_plan", "mobility", "general"):
        return raw
    return "general"


def build_eval_queries(
    real_chunks: list[dict],
    max_queries: int,
    *,
    stratify: bool,
    seed: int,
) -> list[dict]:
    rows_flat: list[dict] = []
    buckets: dict[str, list[dict]] = {"study_plan": [], "general": [], "mobility": []}

    for c in real_chunks:
        m = c.get("metadata", {})
        qs = m.get("questions", []) or []
        ctype = _normalize_content_type(m.get("content_type", "general"))
        for q in qs:
            if not isinstance(q, str) or not q.strip():
                continue
            qtxt = q.strip()
            qshape = "short" if len(qtxt.split()) <= 5 else "long"
            row = {
                "query": qtxt,
                "target_key": _doc_key(c),
                "target_type": ctype,
                "query_shape": qshape,
            }
            rows_flat.append(row)
            buckets[ctype].append(row)

    if not rows_flat:
        raise ValueError("No evaluation questions found in real chunks metadata.")

    if not stratify or max_queries <= 3:
        rng = random.Random(seed)
        rng.shuffle(rows_flat)
        return rows_flat[:max_queries]

    rng = random.Random(seed)
    order = ["study_plan", "general", "mobility"]
    for t in order:
        rng.shuffle(buckets.get(t, []))

    rows: list[dict] = []
    ptr = {t: 0 for t in order}
    while len(rows) < max_queries:
        progressed = False
        for t in order:
            bl = buckets.get(t, [])
            if ptr[t] < len(bl):
                rows.append(bl[ptr[t]])
                ptr[t] += 1
                progressed = True
                if len(rows) >= max_queries:
                    break
        if not progressed:
            break

    if len(rows) < max_queries:
        used_ids = {id(r) for r in rows}
        rest = [r for r in rows_flat if id(r) not in used_ids]
        rng.shuffle(rest)
        for r in rest:
            if len(rows) >= max_queries:
                break
            rows.append(r)

    return rows[:max_queries]


def print_query_mix(queries: list[dict]) -> None:
    by_t: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for q in queries:
        by_t[q["target_type"]] = by_t.get(q["target_type"], 0) + 1
        by_shape[q["query_shape"]] = by_shape.get(q["query_shape"], 0) + 1
    print(f"  Query mix by type   : {by_t}")
    print(f"  Query mix by shape  : {by_shape}")


def corpus_field_text(chunk: dict, field: str) -> str:
    if field == "page":
        return str(chunk.get("page_content", ""))
    if field == "bm25":
        return str(chunk.get("bm25_text") or chunk.get("page_content", ""))
    return str(chunk.get("enriched_content") or chunk.get("page_content", ""))


def embed_passages(model: SentenceTransformer, corpus_texts: list[str], batch_size: int = 128):
    return model.encode(
        ["passage: " + t[:2048].replace("\n", " ") for t in corpus_texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_queries(model: SentenceTransformer, query_texts: list[str], batch_size: int = 128):
    return model.encode(
        ["query: " + q for q in query_texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def compute_metrics(
    queries: list[dict],
    corpus: list[dict],
    model: SentenceTransformer,
    corpus_field: str,
    ks=(1, 3, 5, 10),
    top_k_scan: int = 200,
):
    corpus_texts = [corpus_field_text(c, corpus_field) for c in corpus]
    corpus_keys = [_doc_key(c) for c in corpus]
    query_texts = [row["query"] for row in queries]

    print(f"Embedding corpus ({len(corpus_texts)}) [field={corpus_field}]...")
    doc_vecs = embed_passages(model, corpus_texts)
    print(f"Embedding queries ({len(query_texts)})...")
    qry_vecs = embed_queries(model, query_texts)

    print("Computing similarity matrix...")
    sim_mat = np.dot(qry_vecs, doc_vecs.T)

    hits = {k: 0 for k in ks}
    rr_total = 0.0
    ndcg_total = 0.0
    ranks = []

    for i, row in enumerate(tqdm(queries, desc="Scoring queries")):
        target = row["target_key"]
        rank = None
        sims = sim_mat[i]
        kscan = min(max(1, top_k_scan), len(sims))
        top_idx = np.argpartition(-sims, kscan - 1)[:kscan]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        for pos, idx in enumerate(top_idx, start=1):
            if corpus_keys[idx] == target:
                rank = pos
                break

        if rank is None and kscan < len(sims):
            full_idx = np.argsort(-sims)
            for pos, idx in enumerate(full_idx, start=1):
                if corpus_keys[idx] == target:
                    rank = pos
                    break

        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None:
            rr_total += 1.0 / rank
            ndcg_total += 1.0 / np.log2(rank + 1)
        ranks.append({
            "rank": rank,
            "target_type": row.get("target_type", "unknown"),
            "query_shape": row.get("query_shape", "unknown"),
        })

    n = len(queries)
    return {
        "queries": n,
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr": rr_total / n,
        "ndcg": ndcg_total / n,
    }, ranks


def metrics_from_ranks(rows: list[dict], ks=(1, 3, 5, 10)) -> dict:
    if not rows:
        return {"queries": 0, **{f"hit@{k}": 0.0 for k in ks}, "mrr": 0.0, "ndcg": 0.0}
    hits = {k: 0 for k in ks}
    rr = 0.0
    nd = 0.0
    for r in rows:
        rank = r["rank"]
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None:
            rr += 1.0 / rank
            nd += 1.0 / np.log2(rank + 1)
    n = len(rows)
    return {"queries": n, **{f"hit@{k}": hits[k] / n for k in ks}, "mrr": rr / n, "ndcg": nd / n}


def print_metric_block(title: str, m: dict):
    print(f"\n{title}:")
    for k, v in m.items():
        if k == "queries":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")


def print_segment_delta(label: str, seg: str, m1: dict, m2: dict):
    print(f"\n{label}={seg} (n={m1['queries']}):")
    for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg"]:
        print(f"  {k}: {m2[k] - m1[k]:+.4f}")


def bootstrap_delta(
    r1: list[dict],
    r2: list[dict],
    n_boot: int,
    seed: int,
    ks=(1, 3, 5, 10),
) -> None:
    if n_boot <= 0 or not r1:
        return
    rng = np.random.default_rng(seed)
    n = len(r1)
    keys = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg"]
    deltas = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s1 = [r1[i] for i in idx]
        s2 = [r2[i] for i in idx]
        m1b = metrics_from_ranks(s1, ks=ks)
        m2b = metrics_from_ranks(s2, ks=ks)
        for k in keys:
            deltas[k].append(m2b[k] - m1b[k])
    print(f"\nBootstrap delta (V2-V1), n={n_boot}, seed={seed} — mean [2.5%, 97.5%]:")
    for k in keys:
        arr = np.array(deltas[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {k}: {arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")


def intrinsic_synthetic_quality(
    path: Path,
    model: SentenceTransformer,
    *,
    max_embed: int = 400,
    near_dup_threshold: float = 0.93,
) -> None:
    chunks = load_chunks(path)
    synth = [c for c in chunks if c["metadata"].get("is_synthetic")]
    if not synth:
        print(f"\n[Intrinsic] {path.name}: no synthetic chunks.")
        return

    rng = random.Random(42)
    sample = synth if len(synth) <= max_embed else rng.sample(synth, max_embed)
    texts = [corpus_field_text(c, "enriched") for c in sample]
    print(f"\n[Intrinsic] {path.name}: n_synth={len(synth)}, embed_sample={len(sample)}")
    vecs = embed_passages(model, texts, batch_size=128)
    sim = np.dot(vecs, vecs.T)
    np.fill_diagonal(sim, -1.0)
    max_nei = sim.max(axis=1)
    frac_near_dup = float(np.mean(max_nei > near_dup_threshold))

    frs = [
        float(c["metadata"]["factual_recall"])
        for c in synth
        if c.get("metadata", {}).get("factual_recall") is not None
    ]
    sims_meta = [
        float(c["metadata"].get("filter_sim_score") or c["metadata"].get("similarity_score") or 0)
        for c in synth
        if c.get("metadata", {}).get("filter_sim_score") is not None
        or c.get("metadata", {}).get("similarity_score") is not None
    ]
    print(f"  E5 near-dup rate (max neighbor > {near_dup_threshold}): {frac_near_dup:.3f}")
    if frs:
        print(f"  factual_recall (meta): mean={np.mean(frs):.3f} min={np.min(frs):.3f}")
    if sims_meta:
        print(f"  filter_sim (meta): mean={np.mean(sims_meta):.3f} std={np.std(sims_meta):.3f}")

    by_style: dict[str, list[float]] = {}
    for c in synth:
        st = c.get("metadata", {}).get("synthetic_style", "?")
        sc = c.get("metadata", {}).get("filter_sim_score") or c.get("metadata", {}).get("similarity_score")
        if sc is not None:
            by_style.setdefault(st, []).append(float(sc))
    if by_style:
        print("  filter_sim by style (mean):")
        for st in sorted(by_style.keys()):
            v = by_style[st]
            print(f"    {st}: {np.mean(v):.3f} (n={len(v)})")


def main():
    p = argparse.ArgumentParser(
        description="A/B dense retrieval (E5) + optional intrinsic synthetic quality. "
        "Use --real both (default) for stratified queries across IIT+ISB."
    )
    p.add_argument(
        "--real",
        default="both",
        help="both | comma-separated JSON paths (default loads IIT+ISB augmented)",
    )
    p.add_argument("--v1", required=True, help="Corpus A (e.g. baseline synthetic merge)")
    p.add_argument("--v2", required=True, help="Corpus B (e.g. new synthetic merge)")
    p.add_argument("--max-queries", type=int, default=300)
    p.add_argument("--top-k-scan", type=int, default=200)
    p.add_argument(
        "--corpus-field",
        choices=["enriched", "page", "bm25"],
        default="enriched",
        help="Which chunk field to embed as passage (default: enriched_content)",
    )
    p.add_argument("--no-stratify", action="store_true", help="Disable round-robin stratification by content_type")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples for delta CI (0=off)")
    p.add_argument("--intrinsic-quality", action="store_true", help="Print E5 redundancy + meta stats per corpus")
    args = p.parse_args()

    field_map = {"enriched": "enriched", "page": "page", "bm25": "bm25"}
    corpus_field = field_map[args.corpus_field]

    real_chunks = load_real_chunks(args.real)
    v1_path = Path(args.v1)
    v2_path = Path(args.v2)
    if not v1_path.is_absolute():
        v1_path = PROJECT_ROOT / v1_path
    if not v2_path.is_absolute():
        v2_path = PROJECT_ROOT / v2_path

    v1_chunks = load_chunks(v1_path)
    v2_chunks = load_chunks(v2_path)

    queries = build_eval_queries(
        real_chunks,
        args.max_queries,
        stratify=not args.no_stratify,
        seed=args.seed,
    )
    print(f"\nReal chunks loaded: {len(real_chunks)} | Queries: {len(queries)}")
    print_query_mix(queries)

    model = SentenceTransformer("intfloat/multilingual-e5-large")

    m1, r1 = compute_metrics(
        queries, v1_chunks, model, corpus_field=corpus_field, top_k_scan=args.top_k_scan
    )
    m2, r2 = compute_metrics(
        queries, v2_chunks, model, corpus_field=corpus_field, top_k_scan=args.top_k_scan
    )

    print("\n=== RAG Retrieval A/B (E5) ===")
    print(f"Corpus field: {corpus_field}")
    print(f"Queries: {m1['queries']}")
    print_metric_block("V1", m1)
    print_metric_block("V2", m2)

    print("\nDelta (V2 - V1):")
    for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg"]:
        print(f"  {k}: {m2[k] - m1[k]:+.4f}")

    for t in ["study_plan", "general", "mobility"]:
        g1 = metrics_from_ranks([x for x in r1 if x["target_type"] == t])
        g2 = metrics_from_ranks([x for x in r2 if x["target_type"] == t])
        print_segment_delta("type", t, g1, g2)

    for qshape in ["short", "long"]:
        g1 = metrics_from_ranks([x for x in r1 if x["query_shape"] == qshape])
        g2 = metrics_from_ranks([x for x in r2 if x["query_shape"] == qshape])
        print_segment_delta("query_shape", qshape, g1, g2)

    bootstrap_delta(r1, r2, args.bootstrap, args.seed + 1)

    if args.intrinsic_quality:
        intrinsic_synthetic_quality(v1_path, model)
        intrinsic_synthetic_quality(v2_path, model)


if __name__ == "__main__":
    main()
