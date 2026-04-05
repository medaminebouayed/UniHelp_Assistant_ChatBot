"""
synthetic_chunks.py — Synthetic Data Augmentation for IIT/ISB Graph RAG
=========================================================================
Location : utils/augmentation/synthetic_chunks.py
Run from : project root

    py utils/augmentation/synthetic_chunks.py --count 50 --output data/synthetic/test.json
    py utils/augmentation/synthetic_chunks.py --count 1000 --output data/synthetic/syn_1000.json
    py utils/augmentation/synthetic_chunks.py --count 2000 --merge --output data/synthetic/full.json
    py utils/augmentation/synthetic_chunks.py --count 0 --stats --output data/synthetic/full.json
    py utils/augmentation/synthetic_chunks.py --count 200 --type mobility --noise --output data/synthetic/mobility.json

Improvements in this version:
  IMP 1 — QA answer storage: extract_qa_pairs() stores structured {"q":..., "a":...}
           pairs in metadata["qa_pairs"]. Flat metadata["questions"] kept for BM25.

  IMP 2 — Dynamic similarity thresholds per style:
           reformulate (0.72–0.95) | qa (0.65–0.90) | student (0.60–0.88)
           structured  (0.68–0.92) | query (0.50–0.82)
           Each style has the right window for its expected semantic drift.

  IMP 3 — Noise injection (optional, --noise flag):
           Applies to query-style chunks only. Simulates realistic Tunisian student
           typos to improve retrieval recall on noisy queries.

  All previous fixes retained:
  FIX 1 — Style-aware question extraction
  FIX 2 — Question length cap (120 chars)
  FIX 3 — Duplicate breadcrumb stripping
  FIX 4 — enriched_content != page_content
  FIX 5 — query page_content = source text, queries go to metadata
  + checkpoint/resume, infinite loop protection, lazy model loading

Dependencies:
    pip install requests tqdm python-dotenv sentence-transformers scikit-learn
"""

import os
import sys
import json
import re
import time
import random
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from dotenv import load_dotenv


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

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SOURCE_TOKENS = 80
DEDUP_THRESHOLD   = 0.95
MAX_PER_SOURCE    = 5
MAX_QUESTION_LEN  = 120
MAX_ANSWER_LEN    = 400

# IMP 2 — Dynamic similarity thresholds per style
# (sim_min, sim_max)
SIM_THRESHOLDS: dict[str, tuple[float, float]] = {
    "query"      : (0.50, 0.82),   # short queries drift more from source
    "qa"         : (0.65, 0.90),   # answers grounded, moderate similarity
    "reformulate": (0.72, 0.95),   # paraphrase, stays close to source
    "student"    : (0.60, 0.88),   # casual phrasing, more drift allowed
    "structured" : (0.68, 0.92),   # reorganised, moderate drift
}

_embed_model = None


# ════════════════════════════════════════════════════════════════════════
# LAZY EMBEDDING
# ════════════════════════════════════════════════════════════════════════

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Embedding model ready.")
    return _embed_model


def embed(text: str):
    return get_embed_model().encode([text[:512]])[0]


def cosine(a, b) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    return float(cosine_similarity([a], [b])[0][0])


# ════════════════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════════════════

PROMPT_QUERY = """Génère exactement 5 requêtes courtes qu'un étudiant tunisien taperait sur Google.
Règles STRICTES:
- Une requête par ligne, commençant par "- "
- Maximum 8 mots par requête
- Style naturel, peut avoir des fautes
- RIEN D'AUTRE que les 5 lignes

CONTENU SOURCE:
{source_text}

REQUÊTES (5 lignes, format: - requête):"""

PROMPT_QA = """Transforme ce contenu en 3-4 paires Question/Réponse.
Règles STRICTES:
- Format EXACT: Q: [question courte, max 15 mots]\nR: [réponse complète basée sur le texte]
- Questions formulées comme un étudiant tunisien
- Réponses basées UNIQUEMENT sur le texte source, pas d'invention
- Alterne strictement: Q: puis R: puis Q: puis R:

CONTEXTE: {context_hint}

CONTENU:
{source_text}

PAIRES Q/R:
Q: """

PROMPT_REFORMULATE = """Réécris ce texte universitaire (IIT/ISB Tunisie).
Règles:
- Conserve TOUTES les informations (codes, coûts, noms, dates)
- Structure et vocabulaire complètement différents
- Commence DIRECTEMENT par le texte réécrit, sans titre ni introduction
- Ne répète PAS l'en-tête [IIT|ISB...] du texte original

CONTEXTE: {context_hint}

TEXTE ORIGINAL:
{source_text}

TEXTE RÉÉCRIT (commence directement):"""

PROMPT_STUDENT = """Tu es un étudiant tunisien qui explique à un ami ce qu'il a trouvé.
Règles:
- Style conversationnel en français
- Garde TOUS les détails factuels (codes, coûts, noms, dates)
- Commence DIRECTEMENT par l'explication, sans titre ni introduction
- Ne répète PAS l'en-tête [IIT|ISB...] du texte original

CONTEXTE: {context_hint}

INFORMATIONS:
{source_text}

EXPLICATION (commence directement):"""

PROMPT_STRUCTURED = """Structure ce contenu universitaire (IIT/ISB) en:
1) Titre descriptif (une ligne)
2) Points clés (3-5 bullets avec "- ")
3) Paragraphe d'explication
Ne pas inventer d'informations.
Commence DIRECTEMENT par le titre, sans introduction.

CONTEXTE: {context_hint}

CONTENU:
{source_text}

FORMAT STRUCTURÉ:"""

STYLES = {
    "query"      : PROMPT_QUERY,
    "qa"         : PROMPT_QA,
    "reformulate": PROMPT_REFORMULATE,
    "student"    : PROMPT_STUDENT,
    "structured" : PROMPT_STRUCTURED,
}

STYLE_WEIGHTS = {
    "query"      : 0.25,
    "qa"         : 0.30,
    "reformulate": 0.25,
    "student"    : 0.10,
    "structured" : 0.10,
}


# ════════════════════════════════════════════════════════════════════════
# IMP 1: QA PAIR EXTRACTOR (questions + answers)
# ════════════════════════════════════════════════════════════════════════

def extract_qa_pairs(text: str) -> list[dict]:
    """
    IMP 1: Extract structured {"q": ..., "a": ...} pairs from QA-style output.
    Handles multi-line answers correctly.
    Returns list of dicts, max 6 pairs.
    """
    pairs: list[dict] = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0

    while i < len(lines) and len(pairs) < 6:
        line = lines[i]

        if line.startswith("Q:"):
            q = line[2:].strip()[:MAX_QUESTION_LEN]
            answer_parts: list[str] = []
            j = i + 1

            # Collect answer lines until next Q: or end
            while j < len(lines):
                next_line = lines[j]
                if next_line.startswith("Q:"):
                    break
                if next_line.startswith("R:"):
                    answer_parts.append(next_line[2:].strip())
                elif answer_parts:
                    # continuation of answer
                    answer_parts.append(next_line)
                j += 1

            a = " ".join(answer_parts).strip()[:MAX_ANSWER_LEN]

            if q:
                pairs.append({"q": q, "a": a})
            i = j
        else:
            i += 1

    return pairs


def extract_questions_flat(text: str, style: str) -> list[str]:
    """
    Flat list of question strings for BM25 and enriched_content.
    For QA style: use extract_qa_pairs() first, this just pulls q strings.
    """
    if style == "query":
        qs = []
        for line in text.splitlines():
            q = line.strip().lstrip("-•").strip()
            if len(q) > 5:
                qs.append(q[:MAX_QUESTION_LEN])
        return qs[:6]

    elif style == "qa":
        pairs = extract_qa_pairs(text)
        return [p["q"] for p in pairs]

    else:
        found = re.findall(r'[^.!?]*\?', text)
        seen, result = set(), []
        for q in found:
            q = q.strip()[:MAX_QUESTION_LEN]
            if len(q) > 10 and q not in seen:
                seen.add(q)
                result.append(q)
        return result[:6]


# ════════════════════════════════════════════════════════════════════════
# IMP 3: NOISE INJECTION (query style only)
# ════════════════════════════════════════════════════════════════════════

def inject_noise(text: str, rate: float = 0.08) -> str:
    """
    Simulate realistic Tunisian student search typos.
    Applies to ~8% of words: random char swap, drop, or repeat.
    Only used on query-style chunks when --noise flag is set.
    """
    words = text.split()
    noised = []
    for w in words:
        if len(w) > 3 and random.random() < rate:
            op = random.choice(["swap", "drop", "repeat"])
            j  = random.randint(0, len(w) - 2)
            if op == "swap":
                w = w[:j] + w[j+1] + w[j] + w[j+2:]
            elif op == "drop":
                w = w[:j] + w[j+1:]
            else:  # repeat
                w = w[:j] + w[j] + w[j:]
        noised.append(w)
    return " ".join(noised)


# ════════════════════════════════════════════════════════════════════════
# BREADCRUMB UTILITIES
# ════════════════════════════════════════════════════════════════════════

_BREADCRUMB_RE     = re.compile(r'^\[.*?\|.*?\].*?>\s*', re.DOTALL)
_MULTI_BREADCRUMB  = re.compile(r'(\[.*?\|.*?\].*?>)\s*(\[.*?\|.*?\].*?>)', re.DOTALL)


def strip_breadcrumb(text: str) -> str:
    """FIX 3: Remove any existing [SITE | TITLE] header from LLM output."""
    return _BREADCRUMB_RE.sub('', text).strip()


# ════════════════════════════════════════════════════════════════════════
# OLLAMA
# ════════════════════════════════════════════════════════════════════════

def call_ollama(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model"  : OLLAMA_MODEL,
                    "prompt" : prompt,
                    "stream" : False,
                    "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 500},
                },
                timeout=90,
            )
            return r.json().get("response", "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [Ollama error] {e}")
                return ""
    return ""


# ════════════════════════════════════════════════════════════════════════
# LOAD + WEIGHT SOURCES
# ════════════════════════════════════════════════════════════════════════

def load_sources(
    files        : list[str],
    content_type : Optional[str] = None,
    min_tokens   : int           = MIN_SOURCE_TOKENS,
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
        if c["metadata"].get("token_count", len(c["page_content"]) // 4) >= min_tokens
        and not c["metadata"].get("is_synthetic", False)
    ]
    if content_type:
        valid = [
            c for c in valid
            if c["metadata"].get("content_type", "general") == content_type
        ]
    print(f"  Valid sources (>={min_tokens} tok, real only): {len(valid)}")
    return valid


def source_weights(chunks: list[dict]) -> list[float]:
    return [
        float(
            c["metadata"].get("token_count", 100)
            + len(c["metadata"].get("keywords", [])) * 5
        )
        for c in chunks
    ]


# ════════════════════════════════════════════════════════════════════════
# BUILD CONTEXT HINT
# ════════════════════════════════════════════════════════════════════════

def build_context_hint(meta: dict) -> str:
    return " | ".join(filter(None, [
        meta.get("section_title", meta.get("header", "")),
        meta.get("summary", "")[:100],
        ", ".join(meta.get("keywords", [])[:6]),
    ])) or "Programme universitaire IIT/ISB Tunisie"


# ════════════════════════════════════════════════════════════════════════
# GENERATE ONE SYNTHETIC CHUNK
# ════════════════════════════════════════════════════════════════════════

def make_synthetic(
    source        : dict,
    style         : str,
    src_emb,
    existing_embs : list,
    use_noise     : bool = False,
) -> tuple[Optional[dict], Optional[object]]:
    """
    Generate one synthetic chunk.
    Returns (chunk_dict, gen_embedding) or (None, None) if rejected.
    """
    meta         = source["metadata"]
    source_text  = source["page_content"]
    context_hint = build_context_hint(meta)

    # Build prompt
    if style in ("reformulate", "student", "structured", "qa"):
        prompt = STYLES[style].format(
            source_text  = source_text[:1000],
            context_hint = context_hint,
        )
    else:  # query
        prompt = STYLES[style].format(source_text=source_text[:800])

    generated = call_ollama(prompt)
    if not generated or len(generated.strip()) < 40:
        return None, None

    # FIX 3: strip duplicate breadcrumb
    generated_clean = strip_breadcrumb(generated)
    if len(generated_clean) < 40:
        generated_clean = generated  # fallback

    # IMP 3: noise injection for query style
    if style == "query" and use_noise and random.random() < 0.20:
        generated_clean = inject_noise(generated_clean)

    # Embedding: for query style embed the queries themselves
    # For other styles embed the generated text
    text_for_embed = generated_clean
    gen_emb = embed(text_for_embed)

    # IMP 2: dynamic similarity threshold per style
    sim_min, sim_max = SIM_THRESHOLDS.get(style, (0.65, 0.92))
    sim = cosine(src_emb, gen_emb)
    if sim < sim_min or sim > sim_max:
        return None, None

    # Deduplication
    if any(cosine(gen_emb, e) > DEDUP_THRESHOLD for e in existing_embs):
        return None, None

    # ── IMP 1: QA pair extraction ────────────────────────────────────────
    qa_pairs: list[dict] = []
    if style == "qa":
        qa_pairs = extract_qa_pairs(generated_clean)

    # Flat questions list (for BM25 and enriched_content)
    questions = extract_questions_flat(generated_clean, style)

    # Build fields
    site          = meta.get("site", "")
    title         = meta.get("title", "")
    section_title = meta.get("section_title", meta.get("header", ""))
    summary       = meta.get("summary", "")
    keywords_list = meta.get("keywords", [])
    keywords_text = " ".join(keywords_list)
    breadcrumb    = f"[{site} | {title}] {section_title} > "

    # FIX 5: query page_content = source text (real content for LLM answers)
    if style == "query":
        page_content = source_text
    else:
        page_content = breadcrumb + generated_clean

    # FIX 4: enriched_content properly built
    questions_text   = " ".join(questions)
    qa_answers_text  = " ".join(p["a"] for p in qa_pairs if p.get("a"))

    enriched_content = " ".join(filter(None, [
        section_title,
        summary,
        questions_text,
        qa_answers_text,    # IMP 1: answers also in enriched for better vector
        keywords_text,
        generated_clean,
    ]))

    bm25_text = " ".join(filter(None, [
        section_title,
        summary,
        keywords_text,
        generated_clean,
        questions_text,
        qa_answers_text,    # IMP 1: answers in BM25 too
    ]))

    h = hashlib.md5(generated_clean.encode()).hexdigest()[:8]

    chunk = {
        "page_content"    : page_content,
        "enriched_content": enriched_content,
        "bm25_text"       : bm25_text,
        "metadata": {
            # Source metadata (for filtering)
            "site"            : site,
            "title"           : title,
            "category"        : meta.get("category", ""),
            "url"             : meta.get("url", ""),
            "header"          : section_title,
            "section_title"   : section_title,
            "content_type"    : meta.get("content_type", "general"),
            "summary"         : summary,
            "keywords"        : keywords_list,
            # Synthetic-specific
            "is_synthetic"    : True,
            "synthetic_style" : style,
            "synthetic_id"    : f"syn_{site}_{style}_{h}",
            "similarity_score": round(float(sim), 3),
            "questions"       : questions,          # flat list for BM25
            "qa_pairs"        : qa_pairs,           # IMP 1: structured Q/A
            "has_noise"       : use_noise and style == "query",
            "chunk_length"    : len(page_content),
            "token_count"     : len(generated_clean) // 4,
        },
    }
    return chunk, gen_emb


# ════════════════════════════════════════════════════════════════════════
# SAVE
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
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════

def generate(
    count            : int,
    output_path      : Path,
    content_type     : Optional[str] = None,
    merge_sources    : bool          = False,
    use_noise        : bool          = False,
    checkpoint_every : int           = 50,
    seed             : int           = 42,
) -> list[dict]:

    random.seed(seed)
    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Source files : {[Path(f).name for f in SOURCE_FILES]}")
    print(f"Output       : {output_path}")
    print(f"Noise inject : {use_noise}")

    if not SOURCE_FILES:
        print("[error] No source files found.")
        return []

    print("\nLoading source chunks...")
    sources = load_sources(SOURCE_FILES, content_type=content_type)
    if not sources:
        print("[error] No valid sources.")
        return []

    weights = source_weights(sources)

    # Resume
    synthetic: list[dict] = []
    existing_embs: list   = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        prev = [c for c in existing if c["metadata"].get("is_synthetic", False)]
        if prev:
            print(f"Resuming: {len(prev)} chunks already done. Re-embedding...")
            synthetic = prev
            for c in tqdm(prev, desc="Re-embedding"):
                existing_embs.append(embed(c["page_content"]))

    print(f"\nPre-computing source embeddings ({len(sources)})...")
    src_embs = [
        embed(c["page_content"])
        for c in tqdm(sources, desc="Source embeddings")
    ]

    style_counts   = {s: 0 for s in STYLES}
    source_usage   : dict[str, int] = {}
    total_attempts = 0
    rejected_total = 0
    MAX_ATTEMPTS   = max(count * 12, 500)

    print(f"\nGenerating {count} synthetic chunks...")
    print(f"Sim thresholds: {SIM_THRESHOLDS}")
    pbar = tqdm(total=count, initial=len(synthetic))

    while len(synthetic) < count:
        total_attempts += 1
        if total_attempts > MAX_ATTEMPTS:
            print(
                f"\n[warn] {total_attempts} attempts, only {len(synthetic)}/{count} accepted "
                f"({rejected_total} rejected). Stopping."
            )
            break

        idx     = random.choices(range(len(sources)), weights=weights, k=1)[0]
        source  = sources[idx]
        src_emb = src_embs[idx]
        sid     = source["metadata"].get("url", f"chunk_{idx}")

        if source_usage.get(sid, 0) >= MAX_PER_SOURCE:
            continue

        style = random.choices(
            list(STYLE_WEIGHTS.keys()),
            weights=list(STYLE_WEIGHTS.values()),
            k=1
        )[0]

        chunk, gen_emb = make_synthetic(
            source, style, src_emb, existing_embs, use_noise=use_noise
        )

        if chunk is None:
            rejected_total += 1
            continue

        synthetic.append(chunk)
        existing_embs.append(gen_emb)
        source_usage[sid] = source_usage.get(sid, 0) + 1
        style_counts[style] += 1
        pbar.update(1)

        if len(synthetic) % checkpoint_every == 0:
            save_output(output_path, synthetic, merge_sources)
            pbar.write(f"  Checkpoint {len(synthetic)}/{count}")

        time.sleep(0.05)

    pbar.close()
    save_output(output_path, synthetic, merge_sources)

    acc_rate = len(synthetic) / max(total_attempts, 1) * 100
    print(f"\nDone. Generated={len(synthetic)} Attempts={total_attempts} "
          f"Acceptance={acc_rate:.1f}% Styles={style_counts}")
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

    real      = [c for c in data if not c["metadata"].get("is_synthetic")]
    synthetic = [c for c in data if c["metadata"].get("is_synthetic")]

    styles: dict[str, int] = {}
    sims: list[float]      = []
    q_counts: list[int]    = []
    qa_pair_counts: list[int] = []
    noised = 0

    for c in synthetic:
        s = c["metadata"].get("synthetic_style", "?")
        styles[s] = styles.get(s, 0) + 1
        sc = c["metadata"].get("similarity_score")
        if sc:
            sims.append(sc)
        q_counts.append(len(c["metadata"].get("questions", [])))
        qa_pair_counts.append(len(c["metadata"].get("qa_pairs", [])))
        if c["metadata"].get("has_noise"):
            noised += 1

    types: dict[str, int] = {}
    for c in data:
        t = c["metadata"].get("content_type", "?")
        types[t] = types.get(t, 0) + 1

    print(f"\n── Stats: {output_path.name} ─────────────────────────────────")
    print(f"  Real              : {len(real)}")
    print(f"  Synthetic         : {len(synthetic)}")
    print(f"  Total             : {len(data)}")
    if data:
        print(f"  Synthetic ratio   : {len(synthetic)/len(data)*100:.1f}%")
    if sims:
        print(f"  Avg sim score     : {sum(sims)/len(sims):.3f}  "
              f"(min {min(sims):.3f} / max {max(sims):.3f})")
    if q_counts:
        filled = sum(1 for q in q_counts if q > 0)
        print(f"  Questions filled  : {filled}/{len(synthetic)} chunks, "
              f"avg {sum(q_counts)/max(len(q_counts),1):.1f}")
    if qa_pair_counts:
        filled_qa = sum(1 for q in qa_pair_counts if q > 0)
        print(f"  QA pairs filled   : {filled_qa}/{len(synthetic)} chunks, "
              f"avg {sum(qa_pair_counts)/max(len(qa_pair_counts),1):.1f} pairs")
    print(f"  Noised chunks     : {noised}")
    print(f"  Styles            : {styles}")
    print(f"  Content types     : {types}")
    print(f"──────────────────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic chunks for IIT/ISB Graph RAG."
    )
    parser.add_argument("--count", type=int, required=True,
                        help="Number of synthetic chunks to generate")
    parser.add_argument("--output", type=str,
                        default=str(SYNTHETIC_DIR / "synthetic_chunks.json"),
                        help="Output JSON path")
    parser.add_argument("--type", type=str, default=None,
                        choices=["study_plan", "mobility", "general"])
    parser.add_argument("--merge", action="store_true",
                        help="Merge with real augmented chunks in output")
    parser.add_argument("--noise", action="store_true",
                        help="Apply noise injection to query-style chunks")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats for existing output file and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    if args.stats:
        print_stats(out_path)
    else:
        generate(
            count            = args.count,
            output_path      = out_path,
            content_type     = args.type,
            merge_sources    = args.merge,
            use_noise        = args.noise,
            checkpoint_every = args.checkpoint_every,
            seed             = args.seed,
        )
        print_stats(out_path)