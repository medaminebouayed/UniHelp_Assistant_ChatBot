"""
synthetic_chunks_6.py — Synthetic Data Augmentation for IIT/ISB Graph RAG
=========================================================================
Location : utils/augmentation/synthetic_chunks_6.py
Run from : project root

Fixes applied vs synthetic_chunks.py:
  FIX 1 — Dynamic dedup threshold (relaxes as dataset grows)
  FIX 2 — Global dedup threshold relaxed (0.93 → 0.95)
  FIX 3 — Factual guard relaxed for qa (0.75→0.60) and inference (0.35→0.50)
  FIX 4 — Content type boost unified (3.5→6.0, uses CONTENT_TYPE_WEIGHT_BOOST)
  FIX 5 — Early-exit retry cap (max_attempts = count * 10)
  FIX 6 — Batch embedding pipeline (retrieval embed in batch, query align in batch)

Run:
    python utils/augmentation/synthetic_chunks_6.py --count 200 --noise --output data/synthetic/test.json
    python utils/augmentation/synthetic_chunks_6.py --count 10000 --merge --noise --checkpoint-every 200 --output data/synthetic/full_10k.json
    python utils/augmentation/synthetic_chunks_6.py --count 20000 --merge --noise --checkpoint-every 200 --max-per-source 12 --output data/synthetic/full_20k.json
    python utils/augmentation/synthetic_chunks_6.py --count 0 --stats --output data/synthetic/full_10k.json
"""

import os
import json
import re
import time
import random
import hashlib
import argparse
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# aiohttp is required for async Ollama calls (FIX P1).
# Install: pip install aiohttp
try:
    import aiohttp as _aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    import requests  # synchronous fallback
    print("[warn] aiohttp not found — pip install aiohttp. Falling back to synchronous requests (slower).")

from concurrent.futures import ThreadPoolExecutor, as_completed


# ════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"]               = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

DATA_DIR      = PROJECT_ROOT / "data"
AUGMENTED_DIR = DATA_DIR / "augmented"
RAW_DIR       = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

SOURCE_FILES: list[str] = []
for name in ["iit_chunks_augmented.json", "isb_chunks_augmented.json"]:
    aug = AUGMENTED_DIR / name
    raw = RAW_DIR / name.replace("_augmented", "")
    if aug.exists():
        SOURCE_FILES.append(str(aug))
    elif raw.exists():
        SOURCE_FILES.append(str(raw))
        print(f"[info] Using raw fallback: {raw.name}")


# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════

OLLAMA_URL        = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# FIX P1: Semaphore limits true Ollama concurrency to avoid queue explosion.
# Ollama serialises internally anyway; >4 concurrent requests rarely helps.
# Set OLLAMA_CONCURRENCY=2 if you see queue timeouts.
# CPU auto-defaults to 1 (higher concurrency on CPU compounds latency, not throughput).
OLLAMA_CONCURRENCY = int(os.getenv(
    "OLLAMA_CONCURRENCY",
    "1" if not __import__('torch').cuda.is_available() else "8",
))

# ── Dual-model architecture ────────────────────────────────────────────
FILTER_MODEL_NAME    = "paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVAL_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE     = 64

# ── Source / chunk limits ──────────────────────────────────────────────
MIN_SOURCE_TOKENS  = 30
MAX_SOURCE_TOKENS  = 200
MAX_PER_SOURCE     = 12

# FIX 3: 720 → 580 chars. Max page_content = 580 + ~180 (breadcrumb) = 760 chars = 190 tok.
MAX_GEN_CHARS      = 580
MIN_GEN_CHARS      = 60

# ── Dedup ──────────────────────────────────────────────────────────────
DEDUP_THRESHOLD        = float(os.getenv("SYN_DEDUP_THRESHOLD", "0.93"))
DEDUP_WINDOW           = 300

# FIX 2: Reduced global dedup strength (0.93 → 0.95)
GLOBAL_DEDUP_THRESHOLD           = float(os.getenv("SYN_GLOBAL_DEDUP_THRESHOLD", "0.95"))
GLOBAL_DEDUP_THRESHOLD_STUDYPLAN = float(os.getenv("SYN_GLOBAL_DEDUP_STUDY", "0.97"))


# FIX 1 — Dynamic dedup threshold: relaxes as dataset grows
def get_dedup_threshold(current_size):
    if current_size < 500:
        return 0.92
    elif current_size < 2000:
        return 0.94
    else:
        return 0.96

# ── Similarity filtering (MiniLM space) ────────────────────────────────
SIM_ABSOLUTE_FLOOR = 0.25

SIM_THRESHOLDS: dict[str, tuple[float, float]] = {
    "query"      : (0.25, 0.92),
    "qa"         : (0.40, 0.93),
    "reformulate": (0.35, 0.93),
    "student"    : (0.30, 0.92),
    "structured" : (0.35, 0.92),
    "inference"  : (0.25, 0.88),
    "comparison" : (0.40, 0.96),
}

# ── Style weights ─────────────────────────────────────────────────────
STYLE_WEIGHTS = {
    "query"      : 0.20,
    "qa"         : 0.28,
    "reformulate": 0.15,
    "student"    : 0.10,
    "structured" : 0.10,
    "inference"  : 0.10,
    "comparison" : 0.07,
}

STUDY_PLAN_STYLE_WEIGHTS = {
    "query"      : 0.10,
    "qa"         : 0.42,
    "reformulate": 0.22,
    "student"    : 0.05,
    "structured" : 0.15,
    "inference"  : 0.03,
    "comparison" : 0.03,
}

STYLE_NUM_PREDICT = {
    "query"      : 120,
    "qa"         : 450,
    "reformulate": 350,
    "student"    : 350,
    "structured" : 350,
    "inference"  : 300,
    "comparison" : 350,
}

STYLE_TEMPERATURES = {
    "qa"     : 0.5,
    "default": 0.75,
}

# ── Content type targets ──────────────────────────────────────────────
CONTENT_TYPE_TARGETS = {
    "mobility"  : 0.52,
    "general"   : 0.30,
    "study_plan": 0.18,
}

CONTENT_TYPE_WEIGHT_DAMPEN = 0.05
# FIX 4 — Unified content type boost (was 3.5, now 6.0)
CONTENT_TYPE_WEIGHT_BOOST  = 6.0

MAX_QUESTION_LEN = 120
MAX_ANSWER_LEN   = 400

ADAPT_THRESHOLD      = 0.50
ADAPT_RELAX          = 0.02
ADAPT_INTERVAL       = 50
ADAPT_MAX_RELAX      = 0.08
ADAPT_CEIL_THRESHOLD = 0.40
ADAPT_CEIL_RAISE     = 0.01

# FIX P1: PARALLEL_WORKERS now controls async Ollama concurrency batch size.
# Each "round" submits this many async requests simultaneously.
PARALLEL_WORKERS     = max(1, int(os.getenv("SYN_PARALLEL_WORKERS", "4")))
PARALLEL_BATCH_MULT  = max(1, int(os.getenv("SYN_PARALLEL_BATCH_MULT", "2")))

_JUNK_STARTS = (
    "alors", "et tu", "et vous", "tu sais", "vous savez",
    "dis-moi", "dis moi", "genre", "enfin", "bref", "bon",
    "vraiment", "c'est", "c est",
)
_QUESTION_WORDS = (
    "quel", "quelle", "quels", "quelles", "qui", "que", "quoi",
    "comment", "pourquoi", "combien", "quand", "où", "est-ce",
    "peut-on", "y a-t-il", "existe", "quelles sont", "quels sont",
)

_filter_model         = None
_retrieval_model      = None
_global_chunk_counter = 0


# ════════════════════════════════════════════════════════════════════════
# DUAL-MODEL EMBEDDING
# ════════════════════════════════════════════════════════════════════════

def _create_st_model(model_name: str):
    """Load a SentenceTransformer model with best available backend.
    
    Priority:
      1. CUDA GPU
      2. ONNX Runtime on CPU (2-4× faster than plain CPU, zero quality loss)
      3. Plain CPU PyTorch
    """
    import torch
    import os as _os
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        model = SentenceTransformer(model_name, device="cuda")
        return model, "cuda"

    torch.set_num_threads(max(1, _os.cpu_count() or 4))

    if _os.environ.get("ST_DISABLE_ONNX", "").lower() not in ("1", "true"):
        for try_backend in (
            lambda: SentenceTransformer(model_name, device="cpu", model_kwargs={"backend": "onnx"}),
            lambda: SentenceTransformer(model_name, device="cpu", backend="onnx"),
        ):
            try:
                model = try_backend()
                return model, "cpu+onnx"
            except Exception:
                continue

    model = SentenceTransformer(model_name, device="cpu")
    return model, "cpu"


def get_filter_model():
    global _filter_model
    if _filter_model is None:
        _filter_model, dev = _create_st_model(FILTER_MODEL_NAME)
        print(f"Filter model (MiniLM) ready on {dev.upper()}")
    return _filter_model


def get_retrieval_model():
    global _retrieval_model
    if _retrieval_model is None:
        _retrieval_model, dev = _create_st_model(RETRIEVAL_MODEL_NAME)
        print(f"Retrieval model (E5) ready on {dev.upper()}")
    return _retrieval_model


def _filter_prefix(text: str) -> str:
    return text[:1024].replace("\n", " ")


def _retrieval_prefix(text: str, is_query: bool = False) -> str:
    p = "query: " if is_query else "passage: "
    return p + text[:2048].replace("\n", " ")


def filter_embed_one(text: str) -> np.ndarray:
    return filter_embed_batch([text])[0]


def filter_embed_batch(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    prefixed = [_filter_prefix(t) for t in texts]
    vecs = get_filter_model().encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vecs, dtype=np.float32)


def retrieval_embed_one(text: str, is_query: bool = False) -> np.ndarray:
    return retrieval_embed_batch([text], is_query=is_query)[0]


def retrieval_embed_batch(texts: list[str], is_query: bool = False,
                          batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    prefixed = [_retrieval_prefix(t, is_query) for t in texts]
    vecs = get_retrieval_model().encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vecs, dtype=np.float32)


# FIX P4: Warm up all models + Ollama to eliminate first-call cold-start latency.
def warmup_all():
    print("Warming up models and Ollama...")
    _ = filter_embed_one("warmup")
    _ = retrieval_embed_one("warmup")
    async def _warmup_ollama():
        payload = {
            "model": OLLAMA_MODEL, "prompt": "warmup", "stream": False,
            "options": {"num_ctx": 64, "num_predict": 1},
        }
        if _AIOHTTP_AVAILABLE:
            async with _aiohttp.ClientSession() as session:
                try:
                    async with session.post(OLLAMA_URL, json=payload, timeout=_aiohttp.ClientTimeout(total=30)) as r:
                        await r.text()
                except Exception:
                    pass
        else:
            try:
                import requests as _req
                _req.post(OLLAMA_URL, json=payload, timeout=30)
            except Exception:
                pass
    asyncio.run(_warmup_ollama())
    print("Warmup done.")


# ════════════════════════════════════════════════════════════════════════
# DEDUP WINDOWS
# ════════════════════════════════════════════════════════════════════════

class DedupWindow:
    """Rolling window of E5 embeddings. O(1) matmul check."""
    def __init__(self, maxlen: int, dim: int = 1024):
        self.maxlen  = maxlen
        self.dim     = dim
        self._buf    = np.zeros((maxlen, dim), dtype=np.float32)
        self._ptr    = 0
        self._filled = 0

    def add(self, vec: np.ndarray) -> None:
        self._buf[self._ptr] = vec.astype(np.float32)
        self._ptr    = (self._ptr + 1) % self.maxlen
        self._filled = min(self._filled + 1, self.maxlen)

    def is_duplicate(self, vec: np.ndarray, threshold: float) -> bool:
        if self._filled == 0:
            return False
        sims = self._buf[:self._filled] @ vec.astype(np.float32)
        return bool(np.max(sims) > threshold)

    def load(self, vecs: np.ndarray) -> None:
        n = min(len(vecs), self.maxlen)
        self._buf[:n] = vecs[-n:].astype(np.float32)
        self._ptr     = n % self.maxlen
        self._filled  = n


try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("[warn] faiss not found — pip install faiss-gpu (or faiss-cpu). Falling back to matrix dedup.")


class GlobalDedupIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._count = 0
        if _FAISS_AVAILABLE:
            try:
                res = _faiss.StandardGpuResources()
                cfg = _faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                self._index = _faiss.GpuIndexFlatIP(res, dim, cfg)
                self._backend = "faiss-gpu"
            except Exception:
                self._index = _faiss.IndexFlatIP(dim)
                self._backend = "faiss-cpu"
        else:
            self._capacity = 1024
            self._mat      = np.zeros((self._capacity, dim), dtype=np.float32)
            self._backend  = "numpy"

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2)
        new_mat = np.zeros((new_cap, self.dim), dtype=np.float32)
        new_mat[:self._count] = self._mat[:self._count]
        self._mat      = new_mat
        self._capacity = new_cap

    def add(self, vec: np.ndarray) -> None:
        v = vec.astype(np.float32).reshape(1, -1)
        if self._backend != "numpy":
            self._index.add(v)
        else:
            self._grow(self._count + 1)
            self._mat[self._count] = v
        self._count += 1

    def load(self, vecs: np.ndarray) -> None:
        if vecs.size == 0:
            return
        v = vecs.astype(np.float32)
        if self._backend != "numpy":
            self._index.add(v)
        else:
            n = len(v)
            self._grow(n)
            self._mat[:n] = v
        self._count = len(v)

    def is_duplicate(self, vec: np.ndarray, threshold: float) -> bool:
        if self._count == 0:
            return False
        v = vec.astype(np.float32).reshape(1, -1)
        if self._backend != "numpy":
            D, _ = self._index.search(v, k=1)
            return bool(D[0][0] > threshold)
        else:
            sims = self._mat[:self._count] @ vec.astype(np.float32)
            return bool(np.max(sims) > threshold)


# ════════════════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════════════════

PROMPT_QUERY = """Génère exactement 5 requêtes qu'un étudiant tunisien taperait sur Google.
Règles STRICTES:
- Une requête par ligne, commençant par "- "
- Entre 5 et 12 mots par requête (pas trop court, pas trop long)
- Inclure des mots-clés spécifiques du texte (noms de programmes, coûts, partenaires)
- Style naturel de recherche Google, peut avoir des fautes
- RIEN D'AUTRE que les 5 lignes

CONTENU SOURCE:
{source_text}

REQUÊTES (5 lignes, format: - requête):"""

PROMPT_QA = """Transforme ce contenu en 3-4 paires Question/Réponse.
Règles STRICTES:
- Format EXACT: Q: [question courte, max 15 mots]\nR: [réponse complète basée sur le texte]
- Questions formulées comme un étudiant tunisien
- Réponses basées UNIQUEMENT sur le texte source — NE PAS INVENTER
- Alterne strictement: Q: puis R: puis Q: puis R:
- EXACTEMENT 3 ou 4 paires, pas moins

CONTEXTE: {context_hint}

CONTENU:
{source_text}

PAIRES Q/R:"""

PROMPT_REFORMULATE = """Explique ce contenu universitaire en utilisant des mots COMPLÈTEMENT DIFFÉRENTS.
Règles STRICTES:
- N'utilise AUCUN mot du texte original (sauf noms propres et codes de modules)
- INTERDICTION ABSOLUE d'inventer : garde TOUS les faits exacts (chiffres, dates, codes, noms)
- Si tu ne connais pas un fait, ne le remplace pas — garde-le tel quel
- Commence DIRECTEMENT, pas de titre
- Maximum 500 caractères

CONTEXTE: {context_hint}

TEXTE ORIGINAL (ne pas copier les mots, mais garder TOUS les faits):
{source_text}

EXPLICATION (mots différents, faits identiques):"""

PROMPT_STUDENT = """Tu es un étudiant tunisien qui CONCLUT et DÉDUIT à partir de cette information.
Règles STRICTES:
- INTERDICTION d'inventer des faits non présents dans le texte
- Dis ce que ça IMPLIQUE pour un étudiant (conséquences, avantages pratiques)
- Garde les chiffres et noms EXACTS — ne pas les modifier
- Utilise: "ça veut dire que...", "en pratique...", "ce qu'il faut retenir..."
- Maximum 500 caractères

CONTEXTE: {context_hint}

INFORMATION SOURCE:
{source_text}

DÉDUCTION PRATIQUE:"""

PROMPT_STRUCTURED = """Transforme en fiche technique en ABSTRAYANT les informations clés.
Format OBLIGATOIRE:
DOMAINE: [1 mot ou expression très courte]
NIVEAU: [niveau académique]
POINTS CLÉS: [3-4 bullets de 5 mots MAX chacun, chiffres exacts conservés]
RÉSUMÉ: [1 phrase de synthèse originale, max 100 caractères]

Règles: chaque bullet = fait précis + chiffre/code si disponible. Max 450 caractères total.
NE PAS INVENTER de chiffres ou de faits absents du texte source.

CONTEXTE: {context_hint}

SOURCE:
{source_text}

FICHE:"""

PROMPT_INFERENCE = """En te basant UNIQUEMENT sur ce contenu, réponds à la question implicite d'un étudiant.
La question implicite est: "Qu'est-ce que ça change pour moi concrètement ?"

Règles STRICTES:
- NE PAS INVENTER : parle uniquement de ce qui est dans le texte source
- Parle des CONSÉQUENCES et IMPLICATIONS pratiques (pas du contenu brut)
- Si le texte dit "frais: 1200€/an", explique ce que ça représente
- Garde les chiffres EXACTS
- Maximum 500 caractères
- Commence directement par la réponse pratique

CONTEXTE: {context_hint}

INFORMATION:
{source_text}

IMPLICATION PRATIQUE:"""

PROMPT_COMPARISON = """Un étudiant tunisien compare les deux écoles IIT et ISB.
En te basant UNIQUEMENT sur cette information, génère une comparaison pratique.

Règles STRICTES:
- Compare ce programme/aspect avec ce qu'offre typiquement l'autre école
- Si le texte parle de IIT → mentionne comment ISB diffère (et vice versa)
- NE PAS INVENTER de chiffres — garde les faits exacts du texte source
- Pour l'autre école, utilise des formulations comme "contrairement à...", "alors que..."
- Structure: avantage principal | point de différence | conseil pratique
- Maximum 500 caractères

CONTEXTE: {context_hint}

INFORMATION SOURCE (une seule école):
{source_text}

COMPARAISON IIT vs ISB:"""

STYLES = {
    "query"      : PROMPT_QUERY,
    "qa"         : PROMPT_QA,
    "reformulate": PROMPT_REFORMULATE,
    "student"    : PROMPT_STUDENT,
    "structured" : PROMPT_STRUCTURED,
    "inference"  : PROMPT_INFERENCE,
    "comparison" : PROMPT_COMPARISON,
}


# ════════════════════════════════════════════════════════════════════════
# QUALITY FILTERS
# ════════════════════════════════════════════════════════════════════════

def is_good_question_strict(q: str) -> bool:
    if len(q) < 20:
        return False
    ql = q.lower().strip()
    if any(ql.startswith(j) for j in _JUNK_STARTS):
        return False
    return any(w in ql for w in _QUESTION_WORDS)


def is_good_query(q: str) -> bool:
    q = q.strip()
    if len(q) < 8:
        return False
    return len(re.sub(r'\W', '', q)) >= 5


# ════════════════════════════════════════════════════════════════════════
# QA PAIR EXTRACTOR
# ════════════════════════════════════════════════════════════════════════

_Q_PATTERNS = re.compile(
    r'(?:^|\n)\s*(?:Q\s*:|Question\s*:|\*\*Q\s*:\*\*|\*\*Question\s*:\*\*)\s*(.+)',
    re.IGNORECASE,
)
_A_PATTERNS = re.compile(
    r'(?:^|\n)\s*(?:R\s*:|Réponse\s*:|A\s*:|Answer\s*:|\*\*R\s*:\*\*)\s*(.+)',
    re.IGNORECASE,
)


def extract_qa_pairs(text: str) -> list[dict]:
    pairs: list[dict] = []
    q_matches = list(_Q_PATTERNS.finditer(text))
    a_matches = list(_A_PATTERNS.finditer(text))

    if q_matches and a_matches:
        for qm in q_matches:
            q_text = qm.group(1).strip()[:MAX_QUESTION_LEN]
            q_pos  = qm.start()
            a_text = ""
            for am in a_matches:
                if am.start() > q_pos:
                    a_text = am.group(1).strip()[:MAX_ANSWER_LEN]
                    break
            if q_text and is_good_question_strict(q_text):
                pairs.append({"q": q_text, "a": a_text})
            if len(pairs) >= 6:
                break

    if not pairs:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        i = 0
        while i < len(lines) and len(pairs) < 6:
            line = lines[i]
            if line.startswith("Q:") or line.lower().startswith("question:"):
                q = re.sub(r'^(?:Q:|question:)\s*', '', line, flags=re.IGNORECASE).strip()[:MAX_QUESTION_LEN]
                answer_parts: list[str] = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if re.match(r'^(?:Q:|question:)', nxt, re.IGNORECASE):
                        break
                    if re.match(r'^(?:R:|réponse:|answer:|a:)', nxt, re.IGNORECASE):
                        answer_parts.append(
                            re.sub(r'^(?:R:|réponse:|answer:|a:)\s*', '', nxt, flags=re.IGNORECASE).strip()
                        )
                    elif answer_parts:
                        answer_parts.append(nxt)
                    j += 1
                a = " ".join(answer_parts).strip()[:MAX_ANSWER_LEN]
                if q and is_good_question_strict(q):
                    pairs.append({"q": q, "a": a})
                i = j
            else:
                i += 1
    return pairs


def get_questions(clean_text: str, style: str, source_meta: dict) -> list[str]:
    seen:   set[str]  = set()
    result: list[str] = []

    def add_q(q: str, strict: bool = True):
        q  = q.strip()[:MAX_QUESTION_LEN]
        ok = is_good_question_strict(q) if strict else is_good_query(q)
        if q and q not in seen and ok:
            seen.add(q)
            result.append(q)

    if style == "query":
        for line in clean_text.splitlines():
            add_q(line.strip().lstrip("-•").strip(), strict=False)
    elif style == "qa":
        for p in extract_qa_pairs(clean_text):
            add_q(p["q"], strict=True)
    else:
        for q in source_meta.get("questions", []):
            add_q(str(q), strict=True)
        for q in re.findall(r'[^.!?\n]{15,}[?]', clean_text):
            add_q(q.strip(), strict=True)

    return result[:6]


# ════════════════════════════════════════════════════════════════════════
# NOISE
# ════════════════════════════════════════════════════════════════════════

def inject_noise(text: str, rate: float = 0.08) -> str:
    words  = text.split()
    noised = []
    for w in words:
        if len(w) > 3 and random.random() < rate:
            op = random.choice(["swap", "drop", "repeat"])
            j  = random.randint(0, len(w) - 2)
            if op == "swap":
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            elif op == "drop":
                w = w[:j] + w[j + 1:]
            else:
                w = w[:j] + w[j] + w[j:]
        noised.append(w)
    return " ".join(noised)


_BC_RE = re.compile(r'^\[.*?\|.*?\].*?>\s*', re.DOTALL)


def strip_breadcrumb(text: str) -> str:
    return _BC_RE.sub('', text).strip()


_MODULE_CODE_RE = re.compile(r"\b[A-Z]{2}\d[-\s]\d{3}\b|\b[A-Z]{2}\d(?:-\d)?\b")
_NUMERIC_FACT_RE = re.compile(
    r"\b\d{1,4}(?:[.,]\d+)?\s?(?:€|dt|dinar|dinars|ans?|mois|semestres?|credits?|crédits?|%)\b|\b20\d{2}\b",
    re.IGNORECASE
)


def _extract_critical_facts(text: str) -> set[str]:
    codes = {re.sub(r"\s+", "", m.group(0).strip().upper()) for m in _MODULE_CODE_RE.finditer(text)}
    nums  = {
        re.sub(r"\s+", "", m.group(0).strip().lower()).replace(",", ".")
        for m in _NUMERIC_FACT_RE.finditer(text)
    }
    return codes | nums


def factual_guard(source_text: str, generated_text: str, style: str,
                  content_type: str = "general") -> tuple[bool, float]:
    if style == "query":
        return True, 1.0
    if content_type == "study_plan":
        return True, 1.0
    src = _extract_critical_facts(source_text)
    if not src:
        return True, 1.0
    gen = _extract_critical_facts(generated_text)
    if style in ("inference", "student", "comparison"):
        year_re = re.compile(r'^20\d{2}$')
        extra = set()
        for f in gen:
            if year_re.match(f):
                y = int(f)
                extra.add(str(y - 1))
                extra.add(str(y + 1))
        gen = gen | extra
    kept   = len(src & gen)
    recall = kept / max(len(src), 1)
    # FIX 3 — Relaxed factual guard thresholds
    min_recall = {
        "qa"         : 0.60,
        "reformulate": 0.75,
        "structured" : 0.70,
        "student"    : 0.40,
        "inference"  : 0.50,
        "comparison" : 0.55,
    }.get(style, 0.70)
    return recall >= min_recall, recall


# ════════════════════════════════════════════════════════════════════════
# OLLAMA — ASYNC (FIX P1) + SYNC FALLBACK
# ════════════════════════════════════════════════════════════════════════

def _make_ollama_payload(prompt: str, style: str) -> dict:
    return {
        "model"  : OLLAMA_MODEL,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "num_ctx"    : 1024,
            "temperature": STYLE_TEMPERATURES.get(style, STYLE_TEMPERATURES["default"]),
            "top_p"      : 0.9,
            "num_predict": STYLE_NUM_PREDICT.get(style, 350),
        },
    }


async def _call_ollama_async_one(
    session  : "_aiohttp.ClientSession",
    semaphore: asyncio.Semaphore,
    prompt   : str,
    style    : str,
    retries  : int = 3,
) -> str:
    """Single async Ollama call with semaphore-limited concurrency and retry."""
    payload = _make_ollama_payload(prompt, style)
    async with semaphore:
        for attempt in range(retries):
            try:
                timeout = _aiohttp.ClientTimeout(total=90)
                async with session.post(OLLAMA_URL, json=payload, timeout=timeout) as r:
                    data = await r.json(content_type=None)
                    return data.get("response", "").strip()
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"    [Ollama async error] {e}")
                    return ""
    return ""


async def call_ollama_batch_async(prompts_styles: list[tuple[str, str]]) -> list[str]:
    """
    FIX P1: Fire all (prompt, style) pairs concurrently via aiohttp.
    Semaphore caps true concurrency at OLLAMA_CONCURRENCY to avoid Ollama queue explosion.
    """
    semaphore = asyncio.Semaphore(OLLAMA_CONCURRENCY)
    async with _aiohttp.ClientSession() as session:
        tasks = [
            _call_ollama_async_one(session, semaphore, prompt, style)
            for prompt, style in prompts_styles
        ]
        return await asyncio.gather(*tasks)


def call_ollama_batch_sync(prompts_styles: list[tuple[str, str]]) -> list[str]:
    """Synchronous fallback when aiohttp is unavailable (slow but functional)."""
    import requests as _req
    results = []
    for prompt, style in prompts_styles:
        payload = _make_ollama_payload(prompt, style)
        for attempt in range(3):
            try:
                r = _req.post(OLLAMA_URL, json=payload, timeout=90)
                results.append(r.json().get("response", "").strip())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [Ollama sync error] {e}")
                    results.append("")
                else:
                    time.sleep(2 ** attempt)
    return results


def call_ollama_batch(prompts_styles: list[tuple[str, str]]) -> list[str]:
    """Dispatch to async or sync depending on aiohttp availability."""
    if _AIOHTTP_AVAILABLE:
        return asyncio.run(call_ollama_batch_async(prompts_styles))
    return call_ollama_batch_sync(prompts_styles)


# ════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER (extracted from make_synthetic for batch pipeline)
# ════════════════════════════════════════════════════════════════════════

def build_context_hint(meta: dict) -> str:
    return " | ".join(filter(None, [
        meta.get("section_title", meta.get("header", "")),
        meta.get("summary", "")[:100],
        ", ".join(meta.get("keywords", [])[:6]),
    ])) or "Programme universitaire IIT/ISB Tunisie"


def build_prompt(source: dict, style: str) -> str:
    meta         = source["metadata"]
    source_text  = source["page_content"]
    context_hint = build_context_hint(meta)
    if style in ("reformulate", "student", "structured", "qa", "inference", "comparison"):
        prompt = STYLES[style].replace("{source_text}", source_text[:800]).replace("{context_hint}", context_hint)
    else:
        prompt = STYLES[style].replace("{source_text}", source_text[:600])
    return prompt


# ════════════════════════════════════════════════════════════════════════
# CHUNK PROCESSOR (post-LLM: filter + build, runs after batch embedding)
# ════════════════════════════════════════════════════════════════════════

def process_generated(
    source        : dict,
    style         : str,
    generated     : str,
    filter_emb    : np.ndarray,  # pre-computed MiniLM embedding of generated
    src_filter_emb: np.ndarray,  # pre-computed MiniLM embedding of source
    current_windows: dict,
    use_noise     : bool,
    chunk_counter : int,
) -> tuple[Optional[dict], Optional[np.ndarray], str]:
    """
    Apply diversity gate, factual guard, query alignment, then build the chunk dict.
    Returns (chunk, retrieval_emb, reject_reason) — chunk is None on rejection.
    retrieval_emb is computed here (still one-by-one, but called only on sim-passing chunks).
    For true batch retrieval embedding, caller batches after this stage.
    """
    meta        = source["metadata"]
    source_text = source["page_content"]

    generated_clean = strip_breadcrumb(generated)
    if len(generated_clean) < MIN_GEN_CHARS:
        generated_clean = generated
    if len(generated_clean) < MIN_GEN_CHARS:
        return None, None, "empty"

    # Truncate to MAX_GEN_CHARS at last sentence boundary
    if len(generated_clean) > MAX_GEN_CHARS:
        cutoff      = generated_clean[:MAX_GEN_CHARS]
        last_period = max(cutoff.rfind('. '), cutoff.rfind('.\n'), cutoff.rfind('! '))
        generated_clean = (
            generated_clean[:last_period + 1] if last_period > MAX_GEN_CHARS // 2 else cutoff
        )

    # ── DIVERSITY GATE (MiniLM) ─────────────────────────────────────────
    sim_min, sim_max = current_windows.get(style, (0.35, 0.88))
    sim_min = max(sim_min, SIM_ABSOLUTE_FLOOR)
    sim     = float(np.dot(src_filter_emb, filter_emb))

    if sim < sim_min:
        return None, None, "floor"
    if sim > sim_max:
        return None, None, "ceiling"

    # ── FACTUAL GUARD ───────────────────────────────────────────────────
    fact_ok, fact_recall = factual_guard(
        source_text, generated_clean, style,
        meta.get("content_type", "general"),
    )
    if not fact_ok:
        return None, None, "factual"

    # ── QUERY-PASSAGE ALIGNMENT (query style only — FIX D) ──────────────
    if style == "query":
        q_text_for_align  = " ".join(generated_clean.replace("- ", " ").split())
        query_align_emb   = retrieval_embed_one(q_text_for_align, is_query=True)
        src_retrieval_emb = retrieval_embed_one(source_text[:500], is_query=False)
        align_score       = float(np.dot(query_align_emb, src_retrieval_emb))
        if align_score < 0.72:
            return None, None, "factual"

    # ── Build chunk ─────────────────────────────────────────────────────
    questions = get_questions(generated_clean, style, meta)
    qa_pairs  : list[dict] = []
    if style == "qa":
        qa_pairs = extract_qa_pairs(generated_clean)

    site          = meta.get("site", "")
    title         = meta.get("title", "")
    section_title = meta.get("section_title", meta.get("header", ""))
    summary       = meta.get("summary", "")
    keywords_list = meta.get("keywords", [])
    keywords_text = " ".join(keywords_list)
    breadcrumb    = f"[{site} | {title}] {section_title} > "
    questions_text = " ".join(questions)

    noise_applied = False
    if use_noise and style == "query" and random.random() < 0.15:
        page_content  = inject_noise(source_text)
        noise_applied = True
    elif style == "query":
        page_content  = source_text
    else:
        page_content  = breadcrumb + generated_clean

    if style == "query":
        enriched_content = " ".join(filter(None, [
            section_title, keywords_text, questions_text, source_text[:500], summary,
        ]))
    else:
        enriched_content = " ".join(filter(None, [
            section_title, keywords_text, generated_clean, summary,
        ]))

    if meta.get("content_type") == "study_plan":
        mod_codes = sorted({
            re.sub(r"\s+", "", m.group(0).upper())
            for m in _MODULE_CODE_RE.finditer(source_text.upper())
        })
        if mod_codes:
            enriched_content = " ".join(mod_codes[:36]) + " " + enriched_content

    bm25_text     = " ".join(filter(None, [section_title, keywords_text, questions_text]))
    retrieval_emb = retrieval_embed_one(enriched_content, is_query=False)
    h = hashlib.md5(f"{generated_clean}{chunk_counter}".encode()).hexdigest()[:12]

    chunk = {
        "page_content"    : page_content,
        "enriched_content": enriched_content,
        "bm25_text"       : bm25_text,
        "metadata": {
            "site"            : site,
            "title"           : title,
            "category"        : meta.get("category", ""),
            "url"             : meta.get("url", ""),
            "header"          : section_title,
            "section_title"   : section_title,
            "content_type"    : meta.get("content_type", "general"),
            "summary"         : summary,
            "keywords"        : keywords_list,
            "is_synthetic"    : True,
            "synthetic_style" : style,
            "synthetic_id"    : f"syn_{site}_{style}_{h}",
            "retrieval_boost" : {
                "has_qa"         : len(qa_pairs) > 0,
                "has_questions"  : len(questions) > 0,
                "is_query_style" : style == "query",
                "is_qa_style"    : style == "qa",
                "is_inference"   : style == "inference",
                "is_comparison"  : style == "comparison",
            },
            "filter_sim_score"  : round(float(sim), 3),
            "similarity_score"  : round(float(sim), 3),
            "factual_recall"    : round(float(fact_recall), 3),
            "questions"         : questions,
            "qa_pairs"          : qa_pairs,
            "has_noise"         : noise_applied,
            "chunk_length"      : len(page_content),
            "token_count"       : len(page_content) // 4,
        },
    }
    return chunk, retrieval_emb, ""


# ════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE / LOAD  (FIX P3)
# ════════════════════════════════════════════════════════════════════════

def ckpt_path(output_path: Path) -> Path:
    return output_path.with_suffix("").with_suffix("") \
        if output_path.suffixes == [".ckpt", ".json"] \
        else output_path.parent / (output_path.stem + ".ckpt.json")


def save_checkpoint(output_path: Path, state: dict) -> None:
    """
    Persist all mutable pipeline state so the run can resume exactly from here.
    State keys mirror the .ckpt.json format shown in full_10k.ckpt.json:
      global_chunk_counter, source_usage, style_counts, style_attempts,
      synth_type_counts, doc_key_usage, llm_calls, total_sub_attempts,
      n_floor, n_ceiling, n_factual, n_dedup, n_doc_limit, n_empty,
      adapt_count, source_resets, current_max_cap, current_windows,
      reject_by_style, reject_by_type, attempt_by_type
    """
    path = ckpt_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_checkpoint(output_path: Path) -> Optional[dict]:
    path = ckpt_path(output_path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    print(f"  Checkpoint loaded: {path.name}")
    print(f"  Resuming from chunk {state.get('global_chunk_counter', 0)}, "
          f"llm_calls={state.get('llm_calls', 0)}")
    return state


def _build_checkpoint(
    global_chunk_counter : int,
    source_usage         : dict,
    style_counts         : dict,
    style_attempts       : dict,
    synth_type_counts    : dict,
    doc_key_usage        : dict,
    llm_calls            : int,
    total_sub_attempts   : int,
    n_floor              : int,
    n_ceiling            : int,
    n_factual            : int,
    n_dedup              : int,
    n_doc_limit          : int,
    n_empty              : int,
    adapt_count          : int,
    source_resets        : int,
    current_max_cap      : int,
    current_windows      : dict,
    reject_by_style      : dict,
    reject_by_type       : dict,
    attempt_by_type      : dict,
) -> dict:
    return {
        "global_chunk_counter": global_chunk_counter,
        "source_usage"        : {str(k): v for k, v in source_usage.items()},
        "style_counts"        : style_counts,
        "style_attempts"      : style_attempts,
        "synth_type_counts"   : synth_type_counts,
        "doc_key_usage"       : doc_key_usage,
        "llm_calls"           : llm_calls,
        "total_sub_attempts"  : total_sub_attempts,
        "n_floor"             : n_floor,
        "n_ceiling"           : n_ceiling,
        "n_factual"           : n_factual,
        "n_dedup"             : n_dedup,
        "n_doc_limit"         : n_doc_limit,
        "n_empty"             : n_empty,
        "adapt_count"         : adapt_count,
        "source_resets"       : source_resets,
        "current_max_cap"     : current_max_cap,
        "current_windows"     : current_windows,
        "reject_by_style"     : reject_by_style,
        "reject_by_type"      : reject_by_type,
        "attempt_by_type"     : attempt_by_type,
    }


# ════════════════════════════════════════════════════════════════════════
# LOAD + WEIGHT SOURCES
# ════════════════════════════════════════════════════════════════════════

def load_sources(
    files        : list[str],
    content_type : Optional[str] = None,
    min_tokens   : int           = MIN_SOURCE_TOKENS,
    max_tokens   : int           = MAX_SOURCE_TOKENS,
) -> list[dict]:
    all_chunks: list[dict] = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            print(f"  [warn] Not found: {fpath}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data if isinstance(data, list) else data.get("chunks", [])
        all_chunks.extend(chunks)
        print(f"  {p.name}: {len(chunks)} chunks")

    valid = [
        c for c in all_chunks
        if min_tokens <= len(c["page_content"]) // 4 <= max_tokens
        and not c["metadata"].get("is_synthetic", False)
    ]
    if content_type:
        valid = [c for c in valid if c["metadata"].get("content_type", "general") == content_type]
    print(f"  Valid sources ({min_tokens}–{max_tokens} tok, real only): {len(valid)}")
    return valid


def source_weights(chunks: list[dict]) -> list[float]:
    return [
        float(c["metadata"].get("token_count", 100) + len(c["metadata"].get("keywords", [])) * 5)
        for c in chunks
    ]


def doc_key_from_chunk(chunk: dict) -> str:
    m     = chunk.get("metadata", {})
    site  = m.get("site", "").strip().lower()
    title = m.get("title", "").strip().lower()
    h     = (m.get("section_title") or m.get("header") or "").strip().lower()
    return f"{site}|{title}|{h}"


def pick_style_for_source(source: dict, style_counts: dict = None, style_targets: dict = None) -> str:
    ct     = source["metadata"].get("content_type", "general")
    base_w = STUDY_PLAN_STYLE_WEIGHTS if ct == "study_plan" else STYLE_WEIGHTS
    if style_counts is not None and style_targets is not None:
        available = {s: w for s, w in base_w.items()
                     if style_counts.get(s, 0) < style_targets.get(s, 9999)}
        if available:
            return random.choices(list(available.keys()), weights=list(available.values()), k=1)[0]
    return random.choices(list(base_w.keys()), weights=list(base_w.values()), k=1)[0]


def compute_type_modifier(current_synth: list[dict]) -> dict[str, float]:
    if not current_synth:
        return {}
    counts: dict[str, int] = {}
    for c in current_synth:
        ct = c["metadata"].get("content_type", "general")
        counts[ct] = counts.get(ct, 0) + 1
    total     = len(current_synth)
    modifiers : dict[str, float] = {}
    for ct, target in CONTENT_TYPE_TARGETS.items():
        actual = counts.get(ct, 0) / total if total > 0 else 0
        if actual > target * 1.08:
            modifiers[ct] = CONTENT_TYPE_WEIGHT_DAMPEN
        elif actual < target * 0.70:
            # FIX 4 — Uses shared CONTENT_TYPE_WEIGHT_BOOST instead of hardcoded 3.5
            boost = CONTENT_TYPE_WEIGHT_BOOST if ct == "study_plan" else 2.0
            modifiers[ct] = boost
        else:
            modifiers[ct] = 1.0
    return modifiers


def parse_target_mix(mix: Optional[str]) -> Optional[dict[str, float]]:
    if not mix:
        return None
    parsed: dict[str, float] = {}
    for part in mix.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in CONTENT_TYPE_TARGETS:
            continue
        try:
            parsed[k] = float(v.strip())
        except ValueError:
            continue
    s = sum(parsed.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in parsed.items()}


# ════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ════════════════════════════════════════════════════════════════════════

def save_output(path: Path, synthetic: list[dict], merge_sources: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge_sources:
        real: list[dict] = []
        for fpath in SOURCE_FILES:
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                chunks = data if isinstance(data, list) else data.get("chunks", [])
                real.extend(c for c in chunks if not c["metadata"].get("is_synthetic"))
        output = real + synthetic
        print(f"  Merged: {len(real)} real + {len(synthetic)} synthetic = {len(output)}")
    else:
        output = synthetic
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════════
# ASYNC CHECKPOINT WORKER
# ════════════════════════════════════════════════════════════════════════

def _write_ckpt_io(output_path: Path, synth_snapshot: list, merge_sources: bool, state: dict):
    """Run checkpoint I/O in a background thread (submitted by _do_checkpoint)."""
    save_output(output_path, synth_snapshot, merge_sources)
    save_checkpoint(output_path, state)


# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# FIX P1+P2+FIX6: Batch async LLM calls then batch-embed results each round.
# ════════════════════════════════════════════════════════════════════════

def generate(
    count            : int,
    output_path      : Path,
    content_type     : Optional[str] = None,
    merge_sources    : bool          = False,
    use_noise        : bool          = False,
    checkpoint_every : int           = 100,
    seed             : int           = 42,
    max_per_source   : int           = MAX_PER_SOURCE,
    target_mix       : Optional[str] = None,
    strict_type_quota: bool          = True,
    max_per_doc_key  : int           = 0,
) -> list[dict]:

    global _global_chunk_counter
    random.seed(seed)

    print(f"\nProject root    : {PROJECT_ROOT}")
    print(f"Source files    : {[Path(f).name for f in SOURCE_FILES]}")
    print(f"Output          : {output_path}")
    print(f"Checkpoint      : {ckpt_path(output_path)}")
    print(f"Filter model    : {FILTER_MODEL_NAME}")
    print(f"Retrieval model : {RETRIEVAL_MODEL_NAME}")
    print(f"Torch device    : {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    print(f"ONNX backend    : {__import__('os').environ.get('ST_DISABLE_ONNX', 'auto (try optimum[onnxruntime])')}")
    print(f"Async Ollama    : {'YES (aiohttp)' if _AIOHTTP_AVAILABLE else 'NO (sync fallback)'}")
    print(f"Ollama concur.  : {OLLAMA_CONCURRENCY}")
    print(f"GLOBAL_DEDUP    : {GLOBAL_DEDUP_THRESHOLD}")
    print(f"MAX_GEN_CHARS   : {MAX_GEN_CHARS}")
    print(f"Noise inject    : {use_noise}")
    print(f"Max per source  : {max_per_source}")
    print(f"Max per doc-key : {max_per_doc_key} (0=unlimited)")

    if not SOURCE_FILES:
        print("[error] No source files found.")
        return []

    print("\nLoading source chunks...")
    sources = load_sources(SOURCE_FILES, content_type=content_type)
    if not sources:
        print("[error] No valid sources.")
        return []

    base_weights = source_weights(sources)

    # FIX P4: Warm up before main loop
    warmup_all()

    print(f"\nBatch-encoding {len(sources)} sources with filter model (MiniLM)...")
    src_filter_embs = filter_embed_batch([c["page_content"] for c in sources])
    print(f"Filter embeddings done. Shape: {src_filter_embs.shape}")

    test_vec      = retrieval_embed_one("test", is_query=False)
    retrieval_dim = len(test_vec)
    dedup_window  = DedupWindow(maxlen=DEDUP_WINDOW, dim=retrieval_dim)
    global_dedup  = GlobalDedupIndex(dim=retrieval_dim)
    synthetic     : list[dict] = []

    # ── Initialise counters ────────────────────────────────────────────
    source_usage     : dict[int, int]  = {}
    doc_key_usage    : dict[str, int]  = {}
    style_counts     = {s: 0 for s in STYLES}
    style_attempts   = {s: 0 for s in STYLES}
    synth_type_counts= {k: 0 for k in CONTENT_TYPE_TARGETS}
    llm_calls        = 0
    total_sub_attempts = 0
    n_empty = n_floor = n_ceiling = n_factual = n_dedup = n_doc_limit = 0
    adapt_count = source_resets = 0
    reject_by_style = {s: {"floor": 0, "ceiling": 0, "factual": 0, "dedup": 0, "empty": 0} for s in STYLES}
    reject_by_type  = {ct: {"floor": 0, "ceiling": 0, "factual": 0, "dedup": 0} for ct in CONTENT_TYPE_TARGETS}
    attempt_by_type = {ct: 0 for ct in CONTENT_TYPE_TARGETS}
    current_max_cap  = max_per_source
    current_windows  = {k: list(v) for k, v in SIM_THRESHOLDS.items()}
    base_lows        = {k: v[0] for k, v in SIM_THRESHOLDS.items()}
    base_highs       = {k: v[1] for k, v in SIM_THRESHOLDS.items()}

    # ── Try to resume from checkpoint ─────────────────────────────────
    ckpt = load_checkpoint(output_path)
    if ckpt:
        source_usage      = {int(k): v for k, v in ckpt.get("source_usage", {}).items()}
        style_counts      = ckpt.get("style_counts", style_counts)
        style_attempts    = ckpt.get("style_attempts", style_attempts)
        synth_type_counts = ckpt.get("synth_type_counts", synth_type_counts)
        doc_key_usage     = ckpt.get("doc_key_usage", doc_key_usage)
        llm_calls         = ckpt.get("llm_calls", 0)
        total_sub_attempts= ckpt.get("total_sub_attempts", 0)
        n_floor           = ckpt.get("n_floor", 0)
        n_ceiling         = ckpt.get("n_ceiling", 0)
        n_factual         = ckpt.get("n_factual", 0)
        n_dedup           = ckpt.get("n_dedup", 0)
        n_doc_limit       = ckpt.get("n_doc_limit", 0)
        n_empty           = ckpt.get("n_empty", 0)
        adapt_count       = ckpt.get("adapt_count", 0)
        source_resets     = ckpt.get("source_resets", 0)
        current_max_cap   = ckpt.get("current_max_cap", max_per_source)
        current_windows   = ckpt.get("current_windows", current_windows)
        reject_by_style   = ckpt.get("reject_by_style", reject_by_style)
        reject_by_type    = ckpt.get("reject_by_type", reject_by_type)
        attempt_by_type   = ckpt.get("attempt_by_type", attempt_by_type)
        _global_chunk_counter = ckpt.get("global_chunk_counter", 0)

    # ── Resume synthetic list from output file ─────────────────────────
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        prev = [c for c in existing if c["metadata"].get("is_synthetic", False)]
        if prev:
            print(f"Resuming: {len(prev)} done. Re-embedding for dedup...")
            synthetic             = prev
            _global_chunk_counter = max(_global_chunk_counter, len(prev))
            resume_texts  = [c.get("enriched_content", c["page_content"]) for c in prev[-DEDUP_WINDOW:]]
            resume_embs   = retrieval_embed_batch(resume_texts, is_query=False)
            dedup_window.load(resume_embs)
            all_prev_texts = [c.get("enriched_content", c["page_content"]) for c in prev]
            all_prev_embs  = retrieval_embed_batch(all_prev_texts, is_query=False)
            global_dedup.load(all_prev_embs)
            print(f"Dedup window loaded ({dedup_window._filled} E5 vectors).")

    target_fractions = parse_target_mix(target_mix) or CONTENT_TYPE_TARGETS.copy()
    target_counts    = {k: int(round(count * target_fractions.get(k, 0.0))) for k in CONTENT_TYPE_TARGETS}
    if sum(target_counts.values()) != count:
        max_k = max(target_counts, key=lambda x: target_counts[x])
        target_counts[max_k] += count - sum(target_counts.values())

    style_targets = {s: max(1, round(count * STYLE_WEIGHTS.get(s, 0.01))) for s in STYLES}
    _style_sum    = sum(style_targets.values())
    if _style_sum != count:
        _max_s = max(style_targets, key=lambda s: style_targets[s])
        style_targets[_max_s] += count - _style_sum

    print(f"\nGenerating {count} synthetic chunks...")
    print(f"Content-type targets: {target_counts}")
    pbar = tqdm(total=count, initial=len(synthetic))

    _ckpt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")

    def _do_checkpoint():
        state = _build_checkpoint(
            _global_chunk_counter, source_usage, style_counts, style_attempts,
            synth_type_counts, doc_key_usage, llm_calls, total_sub_attempts,
            n_floor, n_ceiling, n_factual, n_dedup, n_doc_limit, n_empty,
            adapt_count, source_resets, current_max_cap, current_windows,
            reject_by_style, reject_by_type, attempt_by_type,
        )
        synth_snapshot = synthetic[:]
        _ckpt_pool.submit(_write_ckpt_io, output_path, synth_snapshot, merge_sources, state)
        ct_status = " | ".join(
            f"{ct}={synth_type_counts.get(ct,0)}/{target_counts.get(ct,0)}"
            for ct in CONTENT_TYPE_TARGETS
        )
        type_rej = " | ".join(
            f"{ct[:3]}: fa={reject_by_type.get(ct,{}).get('factual',0)} "
            f"dd={reject_by_type.get(ct,{}).get('dedup',0)}"
            for ct in CONTENT_TYPE_TARGETS
        )
        pbar.write(
            f"  [{len(synthetic)}/{count}] "
            f"acc={len(synthetic)/max(llm_calls,1)*100:.0f}% | "
            f"rej: fl={n_floor} ce={n_ceiling} fa={n_factual} "
            f"dd={n_dedup} em={n_empty} | "
            f"types: {ct_status} | type-rej: {type_rej}"
        )

    try:
        # FIX 5 — Early-exit retry cap to prevent "stuck forever"
        max_attempts = count * 10
        attempts = 0
        while len(synthetic) < count and attempts < max_attempts:
            attempts += 1
            free_indices = [
                i for i in range(len(sources))
                if source_usage.get(i, 0) < current_max_cap
            ]

            # Soft content-type quota: restrict to under-quota types
            # but only if enough sources remain (>3) to avoid starving the pipeline.
            if strict_type_quota and content_type is None:
                needed_types = {
                    t for t, n in synth_type_counts.items()
                    if n < target_counts[t]
                }
                if needed_types:
                    typed = [
                        i for i in free_indices
                        if sources[i]["metadata"].get("content_type", "general") in needed_types
                    ]
                    if len(typed) > 3:
                        free_indices = typed

            if not free_indices:
                current_max_cap = max(current_max_cap + 4, max_per_source * 2)
                source_usage.clear()
                pbar.write(f"  [reset #{source_resets}] Sources exhausted — cleared usage, cap → {current_max_cap}/src.")
                source_resets += 1
                continue

            type_mod = compute_type_modifier(synthetic)
            adj_w = [
                base_weights[i] * type_mod.get(
                    sources[i]["metadata"].get("content_type", "general"), 1.0
                )
                for i in free_indices
            ]

            # ── FIX P1+P2: Build batch of jobs, fire async, batch-embed ──
            batch_size    = min(len(free_indices), PARALLEL_WORKERS * PARALLEL_BATCH_MULT, OLLAMA_CONCURRENCY)
            batch_indices : list[int]   = []
            batch_styles  : list[str]   = []
            batch_counters: list[int]   = []

            available = list(zip(free_indices, adj_w))
            for _ in range(batch_size):
                if not available:
                    break
                pool_idx = [x[0] for x in available]
                pool_w   = [x[1] for x in available]
                idx      = random.choices(pool_idx, weights=pool_w, k=1)[0]
                style    = pick_style_for_source(sources[idx], style_counts, style_targets)
                _global_chunk_counter += 1
                batch_indices.append(idx)
                batch_styles.append(style)
                batch_counters.append(_global_chunk_counter)
                style_attempts[style] = style_attempts.get(style, 0) + 1
                llm_calls += 1
                available = [x for x in available if x[0] != idx]

            total_sub_attempts += len(batch_indices)

            # ── Step 1: Build prompts ──────────────────────────────────
            prompts = [
                build_prompt(sources[batch_indices[i]], batch_styles[i])
                for i in range(len(batch_indices))
            ]

            # ── Step 2: Async Ollama batch (FIX P1) ───────────────────
            responses = call_ollama_batch(list(zip(prompts, batch_styles)))

            # ── Step 3: Clean responses, filter empties ────────────────
            cleaned : list[tuple[int, str, str, int, str]] = []  # (list_idx, style, clean, src_idx, counter)
            for i, (resp, style, src_idx, counter) in enumerate(
                zip(responses, batch_styles, batch_indices, batch_counters)
            ):
                if not resp or len(resp.strip()) < MIN_GEN_CHARS:
                    n_empty += 1
                    reject_by_style[style]["empty"] = reject_by_style[style].get("empty", 0) + 1
                    continue
                clean = strip_breadcrumb(resp)
                if len(clean) < MIN_GEN_CHARS:
                    clean = resp
                cleaned.append((i, style, clean, src_idx, counter))

            if not cleaned:
                continue

            # ── Step 4: Batch filter embed (FIX P2) ───────────────────
            gen_texts       = [c[2] for c in cleaned]
            gen_filter_embs = filter_embed_batch(gen_texts) if gen_texts else np.array([])

            # ── Step 5: Apply diversity gate ──────────────────────────
            sim_passed: list[tuple[int, str, str, int, int, float]] = []
            for k, (list_idx, style, clean, src_idx, counter) in enumerate(cleaned):
                src_ct = sources[src_idx]["metadata"].get("content_type", "general")
                attempt_by_type[src_ct] = attempt_by_type.get(src_ct, 0) + 1

                sim_min, sim_max = current_windows.get(style, (0.35, 0.88))
                sim_min = max(sim_min, SIM_ABSOLUTE_FLOOR)
                sim     = float(np.dot(src_filter_embs[src_idx], gen_filter_embs[k]))

                if sim < sim_min:
                    n_floor += 1
                    reject_by_style[style]["floor"] += 1
                    if src_ct in reject_by_type:
                        reject_by_type[src_ct]["floor"] += 1
                    continue
                if sim > sim_max:
                    n_ceiling += 1
                    reject_by_style[style]["ceiling"] += 1
                    if src_ct in reject_by_type:
                        reject_by_type[src_ct]["ceiling"] += 1
                    continue

                sim_passed.append((list_idx, style, clean, src_idx, counter, sim))

            # ── Step 6: FIX 6 — Batch embedding pipeline ────────────────
            # Phase 6a: Parallel factual guard + chunk field building
            #           (ThreadPoolExecutor for CPU-bound text processing)
            _ppool_inputs = [
                (item, sources[item[3]], use_noise and item[1] == "query" and random.random() < 0.15)
                for item in sim_passed
            ]

            def _build_entry(args):
                item, source, noise_flag = args
                list_idx, style, clean, src_idx, counter, sim = item
                meta = source["metadata"]
                source_text = source["page_content"]
                src_ct = meta.get("content_type", "general")

                if len(clean) > MAX_GEN_CHARS:
                    cutoff = clean[:MAX_GEN_CHARS]
                    last_period = max(cutoff.rfind('. '), cutoff.rfind('.\n'), cutoff.rfind('! '))
                    clean = clean[:last_period + 1] if last_period > MAX_GEN_CHARS // 2 else cutoff

                fact_ok, fact_recall = factual_guard(
                    source_text, clean, style, meta.get("content_type", "general")
                )
                if not fact_ok:
                    return ("reject", style, src_ct)

                questions  = get_questions(clean, style, meta)
                qa_pairs   = []
                if style == "qa":
                    qa_pairs = extract_qa_pairs(clean)

                site          = meta.get("site", "")
                title         = meta.get("title", "")
                section_title = meta.get("section_title", meta.get("header", ""))
                summary       = meta.get("summary", "")
                keywords_list = meta.get("keywords", [])
                keywords_text = " ".join(keywords_list)
                breadcrumb    = f"[{site} | {title}] {section_title} > "
                questions_text= " ".join(questions)

                noise_applied = False
                if noise_flag:
                    page_content  = inject_noise(source_text)
                    noise_applied = True
                elif style == "query":
                    page_content  = source_text
                else:
                    page_content  = breadcrumb + clean

                if style == "query":
                    enriched_content = " ".join(filter(None, [
                        section_title, keywords_text, questions_text, source_text[:500], summary,
                    ]))
                else:
                    enriched_content = " ".join(filter(None, [
                        section_title, keywords_text, clean, summary,
                    ]))

                if meta.get("content_type") == "study_plan":
                    mod_codes = sorted({
                        re.sub(r"\s+", "", m.group(0).upper())
                        for m in _MODULE_CODE_RE.finditer(source_text.upper())
                    })
                    if mod_codes:
                        enriched_content = " ".join(mod_codes[:36]) + " " + enriched_content

                bm25_text = " ".join(filter(None, [section_title, keywords_text, questions_text]))

                entry = (
                    list_idx, style, clean, src_idx, counter, sim, fact_recall,
                    source, src_ct, meta, source_text,
                    site, title, section_title, summary, keywords_list,
                    breadcrumb, questions_text, noise_applied, page_content,
                    enriched_content, bm25_text, questions, qa_pairs,
                )

                query_info = None
                if style == "query":
                    q_text = " ".join(clean.replace("- ", " ").split())
                    query_info = (q_text, source_text[:500])

                return ("accept", entry, query_info)

            factual_passed: list[tuple] = []
            query_align_indices: list[int] = []
            query_texts_to_embed: list[str] = []
            src_texts_to_embed: list[str] = []

            with ThreadPoolExecutor(max_workers=4) as _ppool:
                _futures = [_ppool.submit(_build_entry, inp) for inp in _ppool_inputs]
                for _f in as_completed(_futures):
                    _res = _f.result()
                    if _res[0] == "reject":
                        _, _style, _src_ct = _res
                        n_factual += 1
                        reject_by_style[_style]["factual"] += 1
                        if _src_ct in reject_by_type:
                            reject_by_type[_src_ct]["factual"] += 1
                    else:
                        _, _entry, _qinfo = _res
                        _eidx = len(factual_passed)
                        factual_passed.append(_entry)
                        if _qinfo is not None:
                            query_align_indices.append(_eidx)
                            query_texts_to_embed.append(_qinfo[0])
                            src_texts_to_embed.append(_qinfo[1])

            # Phase 6b: Batch embedding
            if factual_passed:
                enriched_texts = [fp[20] for fp in factual_passed]
                all_retrieval_embs = retrieval_embed_batch(enriched_texts, is_query=False)

                # Batch query alignment if needed
                query_align_scores: dict[int, float] = {}
                if query_texts_to_embed:
                    q_embs = retrieval_embed_batch(query_texts_to_embed, is_query=True)
                    s_embs = retrieval_embed_batch(src_texts_to_embed, is_query=False)
                    for i, idx in enumerate(query_align_indices):
                        query_align_scores[idx] = float(np.dot(q_embs[i], s_embs[i]))

                # Phase 6c: Dedup checks using pre-computed embeddings
                for k, fp_entry in enumerate(factual_passed):
                    (
                        list_idx, style, clean, src_idx, counter, sim, fact_recall,
                        source, src_ct, meta, source_text,
                        site, title, section_title, summary, keywords_list,
                        breadcrumb, questions_text, noise_applied, page_content,
                        enriched_content, bm25_text, questions, qa_pairs,
                    ) = fp_entry

                    # Query-passage alignment check
                    if style == "query":
                        if query_align_scores.get(k, 0) < 0.72:
                            n_factual += 1
                            reject_by_style[style]["factual"] += 1
                            continue

                    retrieval_emb = all_retrieval_embs[k]

                    # Rolling dedup with FIX 1 dynamic threshold
                    threshold = get_dedup_threshold(len(synthetic))
                    if dedup_window.is_duplicate(retrieval_emb, threshold):
                        n_dedup += 1
                        reject_by_style[style]["dedup"] += 1
                        if src_ct in reject_by_type:
                            reject_by_type[src_ct]["dedup"] += 1
                        continue

                    # Global dedup
                    g_thresh = (GLOBAL_DEDUP_THRESHOLD_STUDYPLAN
                                if src_ct == "study_plan" else GLOBAL_DEDUP_THRESHOLD)
                    if global_dedup.is_duplicate(retrieval_emb, g_thresh):
                        n_dedup += 1
                        reject_by_style[style]["dedup"] += 1
                        if src_ct in reject_by_type:
                            reject_by_type[src_ct]["dedup"] += 1
                        continue

                    # Doc-key cap
                    dk = doc_key_from_chunk(source)
                    if max_per_doc_key > 0 and doc_key_usage.get(dk, 0) >= max_per_doc_key:
                        n_doc_limit += 1
                        continue

                    # Content-type soft quota (5% overshoot allowed)
                    ct = meta.get("content_type", "general")
                    if strict_type_quota and content_type is None:
                        if synth_type_counts.get(ct, 0) >= target_counts.get(ct, count) * 1.05:
                            n_doc_limit += 1
                            continue

                    h = hashlib.md5(f"{clean}{counter}".encode()).hexdigest()[:12]
                    chunk = {
                        "page_content"    : page_content,
                        "enriched_content": enriched_content,
                        "bm25_text"       : bm25_text,
                        "metadata": {
                            "site"          : site,
                            "title"         : title,
                            "category"      : meta.get("category", ""),
                            "url"           : meta.get("url", ""),
                            "header"        : section_title,
                            "section_title" : section_title,
                            "content_type"  : ct,
                            "summary"       : summary,
                            "keywords"      : keywords_list,
                            "is_synthetic"  : True,
                            "synthetic_style": style,
                            "synthetic_id"  : f"syn_{site}_{style}_{h}",
                            "retrieval_boost": {
                                "has_qa"        : len(qa_pairs) > 0,
                                "has_questions" : len(questions) > 0,
                                "is_query_style": style == "query",
                                "is_qa_style"   : style == "qa",
                                "is_inference"  : style == "inference",
                                "is_comparison" : style == "comparison",
                            },
                            "filter_sim_score": round(sim, 3),
                            "similarity_score": round(sim, 3),
                            "factual_recall"  : round(float(fact_recall), 3),
                            "questions"       : questions,
                            "qa_pairs"        : qa_pairs,
                            "has_noise"       : noise_applied,
                            "chunk_length"    : len(page_content),
                            "token_count"     : len(page_content) // 4,
                        },
                    }

                    synthetic.append(chunk)
                    dedup_window.add(retrieval_emb)
                    global_dedup.add(retrieval_emb)
                    source_usage[src_idx]  = source_usage.get(src_idx, 0) + 1
                    style_counts[style]    = style_counts.get(style, 0) + 1
                    synth_type_counts[ct]  = synth_type_counts.get(ct, 0) + 1
                    doc_key_usage[dk]      = doc_key_usage.get(dk, 0) + 1
                    pbar.update(1)

                    if len(synthetic) % checkpoint_every == 0:
                        _do_checkpoint()

                    if len(synthetic) >= count:
                        break

            # ── Adaptive threshold relaxation ─────────────────────────
            if llm_calls > 0 and llm_calls % ADAPT_INTERVAL == 0:
                recent_accept = len(synthetic) / llm_calls
                if recent_accept < ADAPT_THRESHOLD:
                    relaxed = False
                    for s in current_windows:
                        srej    = reject_by_style[s]
                        s_total = sum(srej.values())
                        if s_total < 5:
                            continue
                        lo, hi = current_windows[s]
                        if srej.get("ceiling", 0) / s_total > ADAPT_CEIL_THRESHOLD:
                            new_hi = min(min(0.99, base_highs[s] + ADAPT_MAX_RELAX), hi + ADAPT_CEIL_RAISE)
                            if new_hi > hi:
                                current_windows[s] = [lo, new_hi]
                                relaxed = True
                        if srej.get("floor", 0) / s_total > 0.40:
                            new_lo = max(max(SIM_ABSOLUTE_FLOOR, base_lows[s] - ADAPT_MAX_RELAX), lo - ADAPT_RELAX)
                            if new_lo < lo:
                                current_windows[s] = [new_lo, current_windows[s][1]]
                                relaxed = True
                    if relaxed:
                        adapt_count += 1
                        pbar.write(f"  [adapt #{adapt_count}] style-aware thresholds updated")

        if len(synthetic) < count:
            pbar.write(f"  [early exit] Hit max_attempts ({max_attempts}) with {len(synthetic)}/{count} chunks.")

    except KeyboardInterrupt:
        pbar.close()
        if synthetic:
            print(f"\n[interrupt] Saving {len(synthetic)} chunks + checkpoint...")
            _do_checkpoint()
        _ckpt_pool.shutdown(wait=True)
        raise
    else:
        pbar.close()
        _do_checkpoint()
        _ckpt_pool.shutdown(wait=True)

    # ── Final summary ─────────────────────────────────────────────────
    total_rej = n_floor + n_ceiling + n_factual + n_dedup + n_empty + n_doc_limit
    print(f"\nDone. Generated={len(synthetic)}")
    print(f"LLM calls       : {llm_calls}")
    print(f"Accepted        : {len(synthetic)} ({len(synthetic)/max(llm_calls,1)*100:.1f}%)")
    print(f"Rejected total  : {total_rej} ({total_rej/max(llm_calls,1)*100:.1f}%)")
    print(f"  → floor       : {n_floor}  ({n_floor/max(total_rej,1)*100:.0f}% of rej)")
    print(f"  → ceiling     : {n_ceiling}  ({n_ceiling/max(total_rej,1)*100:.0f}% of rej)")
    print(f"  → factual     : {n_factual}  ({n_factual/max(total_rej,1)*100:.0f}% of rej)")
    print(f"  → dedup       : {n_dedup}  ({n_dedup/max(total_rej,1)*100:.0f}% of rej)")
    if n_doc_limit:
        print(f"  → doc_key_cap : {n_doc_limit}  ({n_doc_limit/max(total_rej,1)*100:.0f}% of rej)")
    print(f"  → empty/err   : {n_empty}  ({n_empty/max(total_rej,1)*100:.0f}% of rej)")
    print(f"Source resets   : {source_resets} | Adapt events: {adapt_count}")
    print(f"Type counts     : {synth_type_counts} (targets: {target_counts})")
    print(f"Type vs target  :", end=" ")
    for ct in CONTENT_TYPE_TARGETS:
        got  = synth_type_counts.get(ct, 0)
        tgt  = target_counts.get(ct, 0)
        flag = "✓" if abs(got - tgt) <= max(1, tgt * 0.1) else "⚠"
        att  = attempt_by_type.get(ct, 0)
        acc_rate = got / max(att, 1) * 100
        print(f"{ct}={got}/{tgt}{flag}(att={att},acc={acc_rate:.0f}%)", end="  ")
    print()
    if reject_by_type:
        print("Type rejections :", end=" ")
        for ct in CONTENT_TYPE_TARGETS:
            rbt = reject_by_type.get(ct, {})
            print(f"{ct}: fa={rbt.get('factual',0)} dd={rbt.get('dedup',0)} fl={rbt.get('floor',0)}", end="  ")
        print()
    per_style = {
        s: f"{style_counts[s]}/{style_attempts.get(s,0)} ({style_counts[s]/max(style_attempts.get(s,1),1)*100:.0f}%)"
        for s in STYLES
    }
    print(f"Per-style       : {per_style}")
    return synthetic


# ════════════════════════════════════════════════════════════════════════
# STATS
# ════════════════════════════════════════════════════════════════════════

def print_stats(output_path: Path):
    if not output_path.exists():
        print(f"File not found: {output_path}")
        return
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    real  = [c for c in data if not c["metadata"].get("is_synthetic")]
    synth = [c for c in data if c["metadata"].get("is_synthetic")]

    qa_style = [c for c in synth if c["metadata"].get("synthetic_style") == "qa"]
    qa_pairs = [len(c["metadata"].get("qa_pairs", [])) for c in qa_style]

    styles: dict[str, int] = {}
    sims, q_counts, tok_counts = [], [], []
    noised = inherited = 0

    for c in synth:
        s = c["metadata"].get("synthetic_style", "?")
        styles[s] = styles.get(s, 0) + 1
        sc = c["metadata"].get("filter_sim_score") or c["metadata"].get("similarity_score")
        if sc:
            sims.append(sc)
        qc = len(c["metadata"].get("questions", []))
        q_counts.append(qc)
        tok_counts.append(c["metadata"].get("token_count", 0))
        if c["metadata"].get("has_noise"):
            noised += 1
        if s in ("reformulate", "student", "structured", "inference") and qc > 0:
            inherited += 1

    synth_types: dict[str, int] = {}
    for c in synth:
        t = c["metadata"].get("content_type", "?")
        synth_types[t] = synth_types.get(t, 0) + 1

    over_200 = sum(1 for t in tok_counts if t > 200)

    # Checkpoint stats
    ckpt_file = ckpt_path(output_path)
    ckpt_info = ""
    if ckpt_file.exists():
        with open(ckpt_file, encoding="utf-8") as f:
            ckpt = json.load(f)
        ckpt_info = (
            f"\n  Checkpoint      : {ckpt_file.name}"
            f"\n  LLM calls (ckpt): {ckpt.get('llm_calls', '?')}"
            f"\n  Accept rate     : {len(synth)/max(ckpt.get('llm_calls',1),1)*100:.1f}%"
        )

    print(f"\n── Stats: {output_path.name} ──────────────────────────────────")
    print(f"  Filter model    : {FILTER_MODEL_NAME}")
    print(f"  Retrieval model : {RETRIEVAL_MODEL_NAME}")
    print(f"  GLOBAL_DEDUP    : {GLOBAL_DEDUP_THRESHOLD}")
    print(f"  Real            : {len(real)}")
    print(f"  Synthetic       : {len(synth)}")
    print(f"  Total           : {len(data)}")
    if data:
        print(f"  Synthetic ratio : {len(synth)/len(data)*100:.1f}%")
    if sims:
        avg_sim = sum(sims) / len(sims)
        spread  = max(sims) - min(sims)
        health  = "✓ diverse" if spread > 0.25 else "⚠ compressed"
        print(f"  MiniLM sim      : avg={avg_sim:.3f}  min={min(sims):.3f}  max={max(sims):.3f}  spread={spread:.3f}  {health}")
    if tok_counts:
        print(f"  Token count     : avg={sum(tok_counts)//len(tok_counts)} min={min(tok_counts)} max={max(tok_counts)}")
        if over_200:
            print(f"  Chunks > 200 tok: {over_200}/{len(synth)} ⚠")
    if q_counts:
        filled = sum(1 for q in q_counts if q > 0)
        print(f"  Questions filled: {filled}/{len(synth)} avg {sum(q_counts)/max(len(q_counts),1):.1f}")
    if qa_pairs:
        filled_qa = sum(1 for q in qa_pairs if q > 0)
        print(f"  QA pairs filled : {filled_qa}/{len(qa_style)} QA-style, avg {sum(qa_pairs)/max(len(qa_pairs),1):.1f} pairs")
    print(f"  Noised chunks   : {noised} ({noised/max(len(synth),1)*100:.1f}%)")
    print(f"  Styles          : {styles}")
    if synth:
        print(f"  Content types   : {synth_types}")
        ct_pct = {k: f"{v/len(synth)*100:.1f}%" for k, v in synth_types.items()}
        tgt_pct = {k: f"{v*100:.0f}%" for k, v in CONTENT_TYPE_TARGETS.items()}
        print(f"  Content type %  : {ct_pct}")
        print(f"  Content target% : {tgt_pct}")
    frs = [
        float(c["metadata"]["factual_recall"])
        for c in synth
        if c.get("metadata", {}).get("factual_recall") is not None
    ]
    if frs:
        print(f"  factual_recall  : avg={sum(frs)/len(frs):.3f} min={min(frs):.3f}")
    print(ckpt_info)
    print(f"──────────────────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic chunk generation for IIT/ISB Graph RAG.")
    parser.add_argument("--count",            type=int, required=True)
    parser.add_argument("--output",           type=str, default=str(SYNTHETIC_DIR / "synthetic_chunks.json"))
    parser.add_argument("--type",             type=str, default=None, choices=["study_plan", "mobility", "general"])
    parser.add_argument("--merge",            action="store_true")
    parser.add_argument("--noise",            action="store_true")
    parser.add_argument("--stats",            action="store_true")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-per-source",   type=int, default=MAX_PER_SOURCE)
    parser.add_argument("--target-mix",       type=str, default=None,
                        help="e.g. study_plan=0.18,general=0.30,mobility=0.52")
    parser.add_argument("--no-strict-type-quota", action="store_true")
    parser.add_argument(
        "--max-per-doc-key",
        type=int,
        default=3,
        help="Max synthetic chunks per (site|title|section). 0=unlimited.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    if args.stats:
        print_stats(out_path)
    else:
        generate(
            count=args.count, output_path=out_path, content_type=args.type,
            merge_sources=args.merge, use_noise=args.noise,
            checkpoint_every=args.checkpoint_every, seed=args.seed,
            max_per_source=args.max_per_source,
            target_mix=args.target_mix,
            strict_type_quota=not args.no_strict_type_quota,
            max_per_doc_key=args.max_per_doc_key,
        )
        print_stats(out_path)
