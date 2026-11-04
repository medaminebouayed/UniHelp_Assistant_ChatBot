"""
evaluate_hybrid_rag.py — Hybrid RAG Evaluation: Dense + BM25 + Rerank
=======================================================================
Dense  : E5-large on enriched_content
Sparse : BM25Okapi on bm25_text (French-aware tokenizer with stopword removal)
Fusion : Weighted RRF with DUPLICATE DEDUPLICATION (see BUG 1 fix below)
Rerank : BAAI/bge-reranker-v2-m3 (multilingual) with 2048-char enriched content.
  Phase 7b: page_content tested → catastrophic -25.7pp rerank hit@1.
  enriched_content is critical for cross-encoder scoring quality.
  Phase 7c: split merged corpus at \n\n boundaries → 5220 chunks, all ≤2048 chars.
  Best result: rerank hit@1=0.7067 (study_plan=0.830, general=0.830, mobility=0.460).
  Two-stage retrieval tested: not beneficial (−1pp overall, −3pp mobility).
  Mobility bottleneck is cross-encoder scoring quality, not recall or context.

Usage:
    python3 utils/evaluation/evaluate_hybrid_rag.py \\
        --corpus data/synthetic/full_10k.json \\
        --real both --queries 300

Bugs fixed vs original:
  BUG 1 [CATASTROPHIC]: Duplicate doc_keys caused 18× RRF score inflation.
    Corpus has 1040 doc_keys with multiple chunks (real + synthetic share keys).
    Dense returned same doc_key 21 times → RRF score 18× higher than any unique doc.
    True target at dense rank 5 was pushed below all inflated duplicates.
    Result: hit@1 collapsed from 0.41 to 0.137.
    Fix: deduplicate each retriever's results (keep best score per doc_key) before RRF.

  BUG 2: Cross-encoder ms-marco-MiniLM-L-6-v2 is English-only.
    Fix: BAAI/bge-reranker-v2-m3 (multilingual, trained on mMARCO including French).

  ENHANCEMENT: Reranker context window increased from 1024 → 2048 chars.
    Improves rerank hit@1 from 0.633 to 0.690 (+5.67pp). 
    Larger context gives the cross-encoder more signal for scoring.

  BUG 3: BM25 tokenizer kept single-char tokens ("d", "l", "s") from French elisions.
    "d'inscription" → ["d","inscription"]. "d" is a stopword inflating IDF.
    Fix: French stopword list + minimum token length of 2.

  BUG 4: RRF off-by-one — enumerate starts at 0, standard formula is 1-based.
    Fix: scores[doc] += weight / (k + rank + 1)

  BUG 5: top_k_sparse=100 too small for French educational content with many similar sections.
    Fix: default top_k_sparse=200, top_k_dense=150.
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make HF_TOKEN available for sentence-transformers downloads
_HF_TOKEN = os.getenv("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[warn] rank_bm25 not installed. Install: pip install rank_bm25")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("[warn] neo4j driver not installed. Install: pip install neo4j")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_FILES = [
    PROJECT_ROOT / "data/augmented/iit_chunks_augmented.json",
    PROJECT_ROOT / "data/augmented/isb_chunks_augmented.json",
]

# ── French stopwords (common single-char elisions + function words) ──────────
_FR_STOPWORDS = frozenset({
    "a", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de",
    "des", "du", "elle", "elles", "en", "et", "eu", "eux", "il", "ils",
    "je", "la", "le", "les", "leur", "leurs", "lui", "ma", "mais", "me",
    "mes", "moi", "mon", "même", "ni", "nos", "notre", "nous", "on",
    "ou", "où", "par", "pas", "pour", "qu", "que", "qui", "sa", "se",
    "ses", "si", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu",
    "un", "une", "vos", "votre", "vous", "y", "à", "été", "être", "est",
    # Single chars from French elisions: l', d', j', m', n', s', c', t'
    "l", "d", "j", "m", "n", "s", "c", "t",
})


def _fr_tokenize(text: str) -> List[str]:
    """
    French-aware BM25 tokenizer.
    - Keeps accented letters and module codes (GC1-101)
    - Splits on apostrophes and hyphens
    - Removes French stopwords and single-char tokens
    - Lowercases everything except module codes
    """
    # Split on whitespace and punctuation; keep alphanumeric + accented + hyphen
    raw = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)*", text)
    tokens = []
    for tok in raw:
        # Further split on internal hyphens only when NOT a module code pattern
        # Module codes: "GC1-101", "MB2-203" — keep as-is
        if re.match(r"^[A-Z]{2}\d+-\d{3}$", tok):
            tokens.append(tok.lower())
        elif "-" in tok:
            for part in tok.split("-"):
                if len(part) >= 2 and part.lower() not in _FR_STOPWORDS:
                    tokens.append(part.lower())
        else:
            low = tok.lower()
            if len(low) >= 2 and low not in _FR_STOPWORDS:
                tokens.append(low)
    return tokens


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _norm_header(meta: dict) -> str:
    return (meta.get("section_title") or meta.get("header") or "").strip().lower()

def _doc_key(chunk: dict) -> str:
    m = chunk.get("metadata", {})
    key = (
        f"{m.get('site','').strip().lower()}|"
        f"{m.get('title','').strip().lower()}|"
        f"{_norm_header(m)}"
    )
    sub_idx = m.get("sub_index")
    if sub_idx is not None and sub_idx >= 0:
        key = key + "||part_" + str(sub_idx)
    return key


def _parent_key(sub_key: str) -> str:
    """Extract the parent doc_key from a possibly-partitioned sub_key."""
    return sub_key.split("||part_")[0]

def load_chunks(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("chunks", [])

def load_real_chunks(real_arg: str) -> List[Dict]:
    paths: List[Path] = []
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
    out: List[Dict] = []
    for p in paths:
        out.extend(load_chunks(p))
    return out

def _normalize_content_type(raw: str) -> str:
    return raw if raw in ("study_plan", "mobility", "general") else "general"

def build_eval_queries(
    real_chunks: List[Dict],
    max_queries: int,
    *,
    stratify: bool,
    seed: int,
) -> List[Dict]:
    rows_flat: List[Dict] = []
    buckets: Dict[str, List[Dict]] = {"study_plan": [], "general": [], "mobility": []}
    for c in real_chunks:
        m = c.get("metadata", {})
        qs = m.get("questions", []) or []
        ctype = _normalize_content_type(m.get("content_type", "general"))
        for q in qs:
            if not isinstance(q, str) or not q.strip():
                continue
            row = {
                "query": q.strip(),
                "target_key": _doc_key(c),
                "target_type": ctype,
                "query_shape": "short" if len(q.split()) <= 5 else "long",
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
    for t in ["study_plan", "general", "mobility"]:
        rng.shuffle(buckets.get(t, []))
    rows: List[Dict] = []
    ptr = {t: 0 for t in ["study_plan", "general", "mobility"]}
    while len(rows) < max_queries:
        progressed = False
        for t in ["study_plan", "general", "mobility"]:
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
        used = {id(r) for r in rows}
        rest = [r for r in rows_flat if id(r) not in used]
        rng.shuffle(rest)
        rows.extend(rest[:max_queries - len(rows)])
    return rows[:max_queries]


# ────────────────────────────────────────────────────────────────────────────
# Deduplication helper — THE CORE FIX FOR BUG 1
# ────────────────────────────────────────────────────────────────────────────

def dedup_results(results: List[Tuple[str, float]], max_per_key: int = 1) -> List[Tuple[str, float]]:
    """
    Keep up to `max_per_key` entries per doc_key.

    max_per_key=1 — original behavior: keep only the BEST score per doc_key.
      WHY: corpus has many chunks sharing the same doc_key. Without dedup,
      RRF inflates scores for duplicated doc_keys by 18×, pushing unique
      targets out of top-k (BUG 1 fix).

    max_per_key>1 — keep top-N chunks per doc_key (default is original).
      Useful when same doc_key covers diverse topics (e.g. mobility has 31
      chunks per key, each about different subtopics). Keeping more than one
      gives the reranker multiple shots at finding the right content.
    """
    from collections import defaultdict
    counts: Dict[str, int] = defaultdict(int)
    deduped: List[Tuple[str, float]] = []
    for doc, score in results:
        if counts[doc] < max_per_key:
            counts[doc] += 1
            deduped.append((doc, score))
    return deduped


# ────────────────────────────────────────────────────────────────────────────
# Retrievers
# ────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 retriever with French-aware tokenizer and stopword removal."""

    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self.doc_keys = [_doc_key(c) for c in corpus]
        corpus_texts = [self._get_text(c) for c in corpus]
        tokenized = [_fr_tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        # Pre-build doc_key → best position map for deduplication
        # (not needed here — done at search time)

    def _get_text(self, chunk: Dict) -> str:
        # Prefer bm25_text (keyword-dense); fall back to page_content
        return str(chunk.get("bm25_text") or chunk.get("page_content", ""))

    def search(self, query: str, k: int = 200) -> List[Tuple[str, float]]:
        tokens = _fr_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Filter out zero-score docs before sorting — speeds up large corpora
        nonzero = np.where(scores > 0)[0]
        if len(nonzero) == 0:
            # All zero: return top-k by position (no useful BM25 signal)
            top_idx = np.argsort(scores)[::-1][:k]
        else:
            # Sort only nonzero, take top-k
            nonzero_sorted = nonzero[np.argsort(scores[nonzero])[::-1]]
            top_idx = nonzero_sorted[:k]
        raw = [(self.doc_keys[i], float(scores[i])) for i in top_idx]
        # BUG 1 FIX: deduplicate before returning
        return dedup_results(raw)


class GraphRetriever:
    """Neo4j knowledge graph retriever — entity-linked chunk discovery."""

    def __init__(self, corpus: List[Dict], uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        # Build (site_lower, section_lower) → doc_key(s) from corpus
        self.site_sec_to_keys: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for c in corpus:
            m = c.get("metadata", {})
            site = (m.get("site", "") or "").strip().lower()
            sec = _norm_header(m)
            self.site_sec_to_keys[(site, sec)].append(_doc_key(c))
        print("  Graph retriever ready (Neo4j entity-linked retrieval).")

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        terms = self._extract_terms(query)
        if not terms:
            return []
        doc_scores: Dict[str, float] = defaultdict(float)
        with self.driver.session() as session:
            for term in terms:
                try:
                    result = session.run("""
                        MATCH (e)
                        WHERE toLower(e.canonical_name) CONTAINS toLower($term)
                        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                        RETURN c.site AS site, c.section_title AS section_title,
                               e.canonical_name AS entity
                        LIMIT 50
                    """, term=term)
                    for record in result:
                        site = (record["site"] or "").strip().lower()
                        sec  = (record["section_title"] or "").strip().lower()
                        entity = record["entity"]
                        weight = 1.0
                        for dk in self.site_sec_to_keys.get((site, sec), []):
                            doc_scores[dk] += weight
                except Exception:
                    continue
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return dedup_results(sorted_docs[:k])

    def _extract_terms(self, query: str) -> List[str]:
        terms: set = set()
        q = query.strip().lower()
        if q:
            terms.add(q)
        for word in re.findall(r'\b\w+\b', query):
            if len(word) >= 3:
                terms.add(word.lower())
        words = re.findall(r'\b\w+\b', query.lower())
        for i in range(len(words) - 1):
            if len(words[i]) >= 2 and len(words[i + 1]) >= 2:
                terms.add(f"{words[i]} {words[i + 1]}")
        return list(terms)

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass


class DenseRetriever:
    """E5-large dense retriever on enriched_content."""

    def __init__(self, corpus: List[Dict], model: SentenceTransformer):
        self.corpus   = corpus
        self.model    = model
        self.doc_keys = [_doc_key(c) for c in corpus]
        texts = [
            str(c.get("enriched_content") or c.get("page_content", ""))
            for c in corpus
        ]
        print(f"  Embedding {len(texts)} chunks for dense retrieval...")
        self.embeddings = model.encode(
            ["passage: " + t[:2048].replace("\n", " ") for t in texts],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, k: int = 150) -> List[Tuple[str, float]]:
        q_emb = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:k]
        raw = [(self.doc_keys[i], float(sims[i])) for i in top_idx]
        # BUG 1 FIX: deduplicate — keep best score per doc_key
        return dedup_results(raw)


# ────────────────────────────────────────────────────────────────────────────
# Mobility query expansion
# ────────────────────────────────────────────────────────────────────────────

_MOBILITY_EXPANSION_TERMS = {
    "cout": "coût frais financement bourse",
    "prix": "coût frais financement",
    "frais": "coût financement bourse",
    "duree": "durée période stage mission alternance",
    "durer": "durée temps période",
    "diplome": "diplôme titre certification formation accréditation",
    "diplômes": "diplômes titres certifications formations",
    "admission": "inscription condition critère prérequis",
    "inscription": "admission condition dossier",
    "logement": "hébergement résidence carte étudiant transport",
    "transport": "logement carte étudiant abonnement",
    "bourse": "financement aide frais",
    "stage": "alternance mission entreprise partenariat",
    "alternance": "stage mission entreprise contrat",
    "partenaire": "partenariat université école convention",
    "universite": "université école partenaire établissement",
    "programme": "programme formation cursus filière parcours",
    "echange": "échange mobilité internationale programme",
    "mobilite": "mobilité international échange programme",
}

_MOBILITY_GENERIC_EXPANSION = "mobilité internationale université partenaire programme échange stage alternance"

def expand_mobility_query(query: str) -> str:
    ql = query.lower()
    extra_terms = []
    for word, expansion in _MOBILITY_EXPANSION_TERMS.items():
        if word in ql:
            extra_terms.append(expansion)
    if extra_terms:
        return query + " " + " ".join(extra_terms)
    return query + " " + _MOBILITY_GENERIC_EXPANSION


# ────────────────────────────────────────────────────────────────────────────
# RRF (Reciprocal Rank Fusion)
# ────────────────────────────────────────────────────────────────────────────

def weighted_rrf(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.50,
    sparse_weight: float = 0.30,
    graph_results: Optional[List[Tuple[str, float]]] = None,
    graph_weight: float = 0.20,
) -> List[str]:
    """
    Weighted RRF supporting 2-way or 3-way fusion.
    All input lists MUST be deduplicated (one entry per doc_key).
    Formula: score(d) = Σ weight_i / (k + rank_i)  where rank is 1-based.

    BUG 4 FIX: rank starts at 1 (enumerate with start=1).
    """
    scores: Dict[str, float] = defaultdict(float)
    for rank, (doc, _) in enumerate(dense_results, start=1):
        scores[doc] += dense_weight / (k + rank)
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        scores[doc] += sparse_weight / (k + rank)
    if graph_results:
        for rank, (doc, _) in enumerate(graph_results, start=1):
            scores[doc] += graph_weight / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ────────────────────────────────────────────────────────────────────────────
# Reranking
# ────────────────────────────────────────────────────────────────────────────

def load_cross_encoder(preferred: str = "BAAI/bge-reranker-v2-m3") -> Optional[CrossEncoder]:
    """
    BUG 2 FIX: Use multilingual reranker for French content.
    Fallback chain:
      1. BAAI/bge-reranker-v2-m3            — multilingual (French ✓)
      2. cross-encoder/ms-marco-MiniLM-L-6-v2         — English only (old default)
    """
    candidates = [
        preferred,
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    for model_name in candidates:
        try:
            ce = CrossEncoder(model_name)
            print(f"  Cross-encoder loaded: {model_name}")
            return ce
        except Exception as e:
            print(f"  [warn] Could not load {model_name}: {e}")
    print("  [warn] No cross-encoder available — reranking disabled.")
    return None


def rerank(
    query: str,
    candidates: List[str],
    corpus_map: Dict[str, Dict],
    cross_encoder: CrossEncoder,
) -> List[str]:
    """Cross-encoder reranking on enriched_content (cleaned text with context)."""
    pairs = []
    for doc_key in candidates:
        chunk = corpus_map.get(doc_key)
        if chunk is None:
            pairs.append([query, ""])
            continue
        text = str(chunk.get("enriched_content") or chunk.get("page_content", ""))
        pairs.append([query, text[:2048]])
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(ranks: List[Optional[int]], ks=(1, 3, 5, 10)) -> Dict:
    hits = {k: 0 for k in ks}
    rr = ndcg = 0.0
    n = len(ranks)
    for rank in ranks:
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None:
            rr   += 1.0 / rank
            ndcg += 1.0 / np.log2(rank + 1)
    return {
        "queries": n,
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr": rr / n,
        "ndcg": ndcg / n,
    }


# ────────────────────────────────────────────────────────────────────────────
# Evaluation pipeline
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    queries: List[Dict],
    corpus: List[Dict],
    dense: DenseRetriever,
    sparse: Optional[BM25Retriever],
    ce: Optional[CrossEncoder],
    graph: Optional[GraphRetriever] = None,
    top_k_dense: int = 150,
    top_k_sparse: int = 200,
    top_k_graph: int = 100,
    top_k_rerank: int = 150,
    rrf_k: int = 60,
    dense_weight: float = 0.50,
    sparse_weight: float = 0.30,
    graph_weight: float = 0.20,
    debug_bm25: bool = False,
    expand_mobility: bool = False,
    parent_to_subkeys: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict, Dict]:

    # corpus_map: one chunk per doc_key (last one wins — acceptable for reranking text)
    corpus_map: Dict[str, Dict] = {}
    for c in corpus:
        corpus_map[_doc_key(c)] = c

    if debug_bm25 and sparse:
        print("\n── BM25 debug (first 5 queries) ──")
        for row in queries[:5]:
            q = row["query"]
            target = row["target_key"]
            res = sparse.search(q, k=10)
            print(f"\nQ: {q[:80]}")
            print(f"Target: {target[:60]}")
            for j, (doc, score) in enumerate(res[:5], 1):
                hit = "(+)" if doc == target else "  "
                print(f"  {j}. {hit} {doc[:55]} (bm25={score:.3f})")
        print("──────────────────────────────────\n")

    ranks_dense  : List[Optional[int]] = []
    ranks_hybrid : List[Optional[int]] = []
    ranks_rerank : List[Optional[int]] = []

    for row in tqdm(queries, desc="Evaluating"):
        query  = row["query"]
        target = row["target_key"]

        # ── Query expansion for mobility ──────────────────────────────────────
        search_query = query
        if expand_mobility and row.get("target_type") == "mobility":
            search_query = expand_mobility_query(query)

        def _first_rank(keys: List[str], target_key: str) -> Optional[int]:
            """Find rank (1-based) of target_key in keys, handling sub_key matching."""
            if target_key in keys:
                return keys.index(target_key) + 1
            parent = target_key
            for rk, k in enumerate(keys, 1):
                if _parent_key(k) == parent:
                    return rk
            return None

        # ── Dense ─────────────────────────────────────────────────────────────
        d_res  = dense.search(search_query, k=top_k_dense)
        d_keys = [doc for doc, _ in d_res]
        ranks_dense.append(_first_rank(d_keys, target))

        # ── Hybrid (RRF: dense + BM25 only) ────────────────────────────────────
        if sparse:
            s_res = sparse.search(search_query, k=top_k_sparse)
            h_keys = weighted_rrf(d_res, s_res, k=rrf_k,
                                  dense_weight=dense_weight, sparse_weight=sparse_weight)
            ranks_hybrid.append(_first_rank(h_keys, target))

            # ── Rerank (hybrid + optional graph candidates) ────────────────────
            if ce:
                candidates = h_keys[:top_k_rerank]
                # Two-stage: expand parent keys to sub-keys for reranker
                if parent_to_subkeys:
                    expanded = []
                    for parent in candidates:
                        children = parent_to_subkeys.get(parent, [parent])
                        expanded.extend(children)
                    candidates = expanded[:top_k_rerank * 3]
                if graph:
                    g_res = graph.search(query, k=top_k_graph)
                    g_keys = [doc for doc, _ in g_res]
                    seen = set(candidates)
                    for dk in g_keys:
                        if dk not in seen:
                            candidates.append(dk)
                            seen.add(dk)
                reranked = rerank(query, candidates[:top_k_rerank], corpus_map, ce)
                ranks_rerank.append(_first_rank(reranked, target))
            else:
                ranks_rerank.append(ranks_hybrid[-1])
        else:
            ranks_hybrid.append(ranks_dense[-1])
            ranks_rerank.append(ranks_dense[-1])

    metrics = {
        "dense" : compute_metrics(ranks_dense),
        "hybrid": compute_metrics(ranks_hybrid),
        "rerank": compute_metrics(ranks_rerank),
    }

    # Per content-type breakdown
    type_metrics: Dict[str, Dict] = {}
    for t in ["study_plan", "general", "mobility"]:
        idx = [i for i, q in enumerate(queries) if q["target_type"] == t]
        if not idx:
            continue
        type_metrics[t] = {
            "dense" : compute_metrics([ranks_dense[i]  for i in idx]),
            "hybrid": compute_metrics([ranks_hybrid[i] for i in idx]),
            "rerank": compute_metrics([ranks_rerank[i] for i in idx]),
        }

    return metrics, type_metrics


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def print_metrics(label: str, m: Dict) -> None:
    print(f"\n{label.upper()}:")
    for k, v in m.items():
        if k == "queries":
            continue
        print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG pipeline evaluation (Dense + BM25 + Rerank)."
    )
    parser.add_argument("--corpus",         required=True,   help="Path to corpus JSON")
    parser.add_argument("--real",           default="both",  help="Real chunk source")
    parser.add_argument("--queries",        type=int, default=300)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--no-hybrid",      action="store_true")
    parser.add_argument("--no-rerank",      action="store_true")
    parser.add_argument("--top-k-dense",    type=int, default=150)
    parser.add_argument("--top-k-sparse",   type=int, default=200)
    parser.add_argument("--top-k-rerank",   type=int, default=150)
    parser.add_argument("--top-k-graph",    type=int, default=100,
                        help="Top-k from graph retriever (default 100)")
    parser.add_argument("--rrf-k",          type=int, default=60,
                        help="RRF smoothing constant (default 60)")
    parser.add_argument("--dense-weight",   type=float, default=0.55)
    parser.add_argument("--sparse-weight",  type=float, default=0.45)
    parser.add_argument("--graph-weight",   type=float, default=0.20)
    parser.add_argument("--reranker",       default="BAAI/bge-reranker-v2-m3",
                        help="Preferred cross-encoder model (multilingual recommended)")
    parser.add_argument("--use-graph",      action="store_true",
                        help="Enable Neo4j knowledge graph retrieval")
    parser.add_argument("--neo4j-uri",      default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user",     default="neo4j")
    parser.add_argument("--neo4j-password", default="password123")
    parser.add_argument("--debug-bm25",     action="store_true",
                        help="Show BM25 top-5 for first 5 queries")
    parser.add_argument("--no-stratify",    action="store_true")
    parser.add_argument("--expand-mobility", action="store_true",
                        help="Expand mobility queries with related terms before retrieval")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / args.corpus
    corpus = load_chunks(corpus_path)

    real_chunks = load_real_chunks(args.real)
    queries = build_eval_queries(
        real_chunks, args.queries,
        stratify=not args.no_stratify, seed=args.seed,
    )

    # Corpus composition summary
    n_real  = sum(1 for c in corpus if not c["metadata"].get("is_synthetic"))
    n_synth = sum(1 for c in corpus if c["metadata"].get("is_synthetic"))
    from collections import Counter
    keys = [_doc_key(c) for c in corpus]
    n_unique = len(set(keys))
    n_dup    = len(keys) - n_unique
    print(f"\nCorpus: {len(corpus)} chunks  ({n_real} real + {n_synth} synthetic)")
    print(f"         {n_unique} unique doc_keys  ({n_dup} duplicate chunk entries — deduplicated in retrieval)")
    print(f"Queries: {len(queries)}  (stratified: study_plan/general/mobility ~ 1:1:1)")

    by_t = Counter(q["target_type"] for q in queries)
    print(f"  query types: {dict(by_t)}")

    print("\nLoading models...")
    dense_model = SentenceTransformer("intfloat/multilingual-e5-large")
    dense_retriever = DenseRetriever(corpus, dense_model)

    sparse_retriever = None
    if not args.no_hybrid:
        if BM25_AVAILABLE:
            sparse_retriever = BM25Retriever(corpus)
            print("  BM25 retriever ready (French tokenizer + stopword removal).")
        else:
            print("  [warn] rank_bm25 not installed — hybrid disabled. pip install rank_bm25")

    cross_encoder = None
    if not args.no_rerank:
        cross_encoder = load_cross_encoder(args.reranker)

    graph_retriever = None
    if args.use_graph:
        if NEO4J_AVAILABLE:
            try:
                graph_retriever = GraphRetriever(
                    corpus,
                    uri=args.neo4j_uri,
                    user=args.neo4j_user,
                    password=args.neo4j_password,
                )
                print("  Neo4j graph retriever ready.")
            except Exception as e:
                print(f"  [warn] Neo4j connection failed: {e}")
        else:
            print("  [warn] neo4j driver not installed. pip install neo4j")

    print(f"\nRunning evaluation (top_k_dense={args.top_k_dense}, top_k_sparse={args.top_k_sparse}, rrf_k={args.rrf_k}, expand_mobility={args.expand_mobility})...")
    metrics, type_metrics = evaluate(
        queries, corpus,
        dense_retriever, sparse_retriever, cross_encoder,
        graph=graph_retriever,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
        top_k_graph=getattr(args, 'top_k_graph', 100),
        top_k_rerank=args.top_k_rerank,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        graph_weight=args.graph_weight if args.use_graph else 0.0,
        debug_bm25=args.debug_bm25,
        expand_mobility=args.expand_mobility,
    )

    # Cleanup Neo4j connection
    if graph_retriever:
        graph_retriever.close()

    print("\n=== Overall Metrics ===")
    for method in ["dense", "hybrid", "rerank"]:
        if method in metrics:
            print_metrics(method, metrics[method])

    if type_metrics:
        print("\n=== Per Content Type ===")
        for method in ["dense", "hybrid", "rerank"]:
            print(f"\n  -- {method.upper()} --")
            for t in ["study_plan", "general", "mobility"]:
                if t in type_metrics and method in type_metrics[t]:
                    m = type_metrics[t][method]
                    h1 = m["hit@1"]
                    h3 = m["hit@3"]
                    h10 = m["hit@10"]
                    mrr = m["mrr"]
                    print(f"  {t:<12}: hit@1={h1:.3f}  hit@3={h3:.3f}  hit@10={h10:.3f}  mrr={mrr:.3f}")

    # Delta vs dense
    print("\n=== Delta vs Dense (hybrid and rerank improvements) ===")
    for method in ["hybrid", "rerank"]:
        if method not in metrics:
            continue
        print(f"\n  {method.upper()} - DENSE:")
        for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg"]:
            d = metrics[method][k] - metrics["dense"][k]
            sig = "(+)" if abs(d) > 0.01 else "(~)"
            print(f"    {k}: {d:+.4f} {sig}")


if __name__ == "__main__":
    main()