"""
sparse_index — BM25 index build + search with pickle persistence.

Full rebuild on every index call (fast enough: ~50ms for 5K docs).
"""

import re
import pickle
from typing import List, Tuple, Dict, Optional

from rank_bm25 import BM25Okapi

from rag.config import BM25_INDEX_PATH
from rag.store import get_all_chunks

_FR_STOPWORDS = frozenset({
    "a", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de",
    "des", "du", "elle", "elles", "en", "et", "eu", "eux", "il", "ils",
    "je", "la", "le", "les", "leur", "leurs", "lui", "ma", "mais", "me",
    "mes", "moi", "mon", "meme", "ni", "nos", "notre", "nous", "on",
    "ou", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se",
    "ses", "si", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu",
    "un", "une", "vos", "votre", "vous", "y", "a", "ete", "etre", "est",
    "l", "d", "j", "m", "n", "s", "c", "t",
})


def _fr_tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)*", text)
    tokens = []
    for tok in raw:
        low = tok.lower()
        if len(low) >= 2 and low not in _FR_STOPWORDS:
            tokens.append(low)
    return tokens


def rebuild_index():
    """Rebuild BM25 index from ALL chunks in PostgreSQL and save to pickle."""
    chunks = get_all_chunks()
    print(f"[sparse] Rebuilding BM25 index from {len(chunks)} chunks...")

    doc_keys = []
    tokenized = []

    for c in chunks:
        text = c.get("bm25_text") or c.get("page_content", "") or ""
        doc_keys.append(c["doc_key"])
        tokenized.append(_fr_tokenize(text))

    bm25 = BM25Okapi(tokenized)
    index_data = {
        "bm25": bm25,
        "doc_keys": doc_keys,
        "tokenized": tokenized,
    }

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    global _INDEX_CACHE
    _INDEX_CACHE = index_data

    print(f"[sparse] BM25 index saved to {BM25_INDEX_PATH} ({len(doc_keys)} docs)")


_INDEX_CACHE = None

def _load_index() -> Optional[Dict]:
    """Load BM25 index from pickle, or return cached instance if available."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
        
    if not BM25_INDEX_PATH.exists():
        print("[sparse] No BM25 index found. Run index.py first.")
        return None
    with open(BM25_INDEX_PATH, "rb") as f:
        _INDEX_CACHE = pickle.load(f)
        return _INDEX_CACHE


def sparse_search(query: str, k: int = 200) -> List[Tuple[str, float]]:
    """Search BM25 index. Returns [(doc_key, score)]."""
    index = _load_index()
    if index is None:
        return []
    tokens = _fr_tokenize(query)
    scores = index["bm25"].get_scores(tokens)
    doc_keys = index["doc_keys"]

    top_idx = scores.argsort()[::-1][:k]
    return [(doc_keys[i], float(scores[i])) for i in top_idx if scores[i] > 0]
