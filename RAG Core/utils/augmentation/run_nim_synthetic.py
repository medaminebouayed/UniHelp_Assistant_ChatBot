"""
run_nim_synthetic.py — Mass synthetic generation using NVIDIA NIM.

Generates 2-4 synthetic variants per real chunk via NVIDIA NIM (LLaMA 3.1 8B)
to reach 10k total chunks. Reuses the same prompt templates and quality gates
as synthetic_chunks.py but calls NIM's OpenAI-compatible API instead of Ollama.

Usage:
    python run_nim_synthetic.py                     # generate until 10k
    python run_nim_synthetic.py --target 15000       # custom target
    python run_nim_synthetic.py --max-per-source 6   # more variants per chunk
    python run_nim_synthetic.py --dry-run            # just count, don't generate

Requirements:
    pip install numpy tqdm psycopg2-binary sentence-transformers
    pip install httpx    # for NIM HTTP calls

Env vars (.env or export):
    NIM_API_URL  = http://localhost:8000/v1/chat/completions  (self-hosted NIM)
    NIM_API_KEY  = nvapi-...                                  (NVIDIA cloud key)
    NIM_MODEL    = meta/llama-3.1-8b-instruct                 (default)
"""

import os
import re
import json
import sys
import random
import hashlib
import argparse
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from rag.store import get_conn, insert_chunk, ensure_schema

# ── Reuse synthetic_chunks prompts & quality gates ─────────────────────
from utils.augmentation.synthetic_chunks import (
    STYLES, STYLE_WEIGHTS, STUDY_PLAN_STYLE_WEIGHTS,
    SIM_THRESHOLDS, SIM_ABSOLUTE_FLOOR,
    MAX_GEN_CHARS, MIN_GEN_CHARS,
    build_prompt, factual_guard,
    get_questions, extract_qa_pairs,
    strip_breadcrumb, inject_noise,
    filter_embed_one, get_filter_model,
)

# ── NIM Config ──────────────────────────────────────────────────────────

NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
NIM_MODEL   = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# NIM concurrency (NIM handles batching internally, but we limit client-side)
NIM_CONCURRENCY = int(os.getenv("NIM_CONCURRENCY", "8"))

# ── Generation config ──────────────────────────────────────────────────

VARIANTS_PER_CHUNK = 3
MAX_VARIANTS_PER_CHUNK = 4
TARGET_TOTAL_CHUNKS = 10000
STYLE_CHOICES = list(STYLE_WEIGHTS.keys())
NOISE_RATE = 0.08

NUM_PREDICT = {
    "query": 120, "qa": 450, "reformulate": 350, "student": 350,
    "structured": 350, "inference": 300, "comparison": 350,
}
TEMPERATURES = {"qa": 0.5, "default": 0.75}


# ════════════════════════════════════════════════════════════════════════
# NIM Client
# ════════════════════════════════════════════════════════════════════════

def _nim_payload(prompt: str, style: str) -> dict:
    return {
        "model": NIM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un assistant specialise dans la generation de contenu universitaire "
                           "pour IIT et ISB Tunisie. Reponds en francais uniquement. "
                           "Ne fais que ce qui est demande, sans ajouter de commentaires.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURES.get(style, TEMPERATURES["default"]),
        "top_p": 0.9,
        "max_tokens": NUM_PREDICT.get(style, 350),
    }


def _nim_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if NIM_API_KEY:
        headers["Authorization"] = f"Bearer {NIM_API_KEY}"
    return headers


def _parse_nim_response(resp_data: dict) -> str:
    try:
        return resp_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return ""


async def call_nim_batch(prompts_styles: list[tuple[str, str]]) -> list[str]:
    """Fire multiple prompts to NIM concurrently via httpx."""
    import httpx

    semaphore = asyncio.Semaphore(NIM_CONCURRENCY)

    async def _call_one(prompt: str, style: str) -> str:
        payload = _nim_payload(prompt, style)
        async with semaphore:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        resp = await client.post(
                            NIM_API_URL,
                            json=payload,
                            headers=_nim_headers(),
                        )
                    resp.raise_for_status()
                    text = _parse_nim_response(resp.json())
                    if text:
                        return text
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"    [NIM error] {e}")
                        return ""
        return ""

    tasks = [_call_one(p, s) for p, s in prompts_styles]
    return await asyncio.gather(*tasks)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _pick_styles(content_type: str) -> list[str]:
    weights = STUDY_PLAN_STYLE_WEIGHTS if content_type == "study_plan" else STYLE_WEIGHTS
    available = [s for s in STYLE_CHOICES if s in weights]
    n = random.randint(2, min(MAX_VARIANTS_PER_CHUNK, len(available)))
    if n <= 0:
        return ["qa"]
    chosen = random.choices(available, weights=[weights[s] for s in available], k=n)
    return chosen


def _make_synthetic_id(site: str, style: str, text: str, counter: int) -> str:
    h = hashlib.md5(f"{text}{counter}".encode()).hexdigest()[:12]
    return f"syn_{site}_{style}_{h}"


def _pg_row_to_source(row: dict) -> dict:
    meta = {
        "site": row.get("site", ""),
        "title": row.get("title", ""),
        "header": row.get("section_title", ""),
        "section_title": row.get("section_title", ""),
        "category": row.get("content_type", "general"),
        "content_type": row.get("content_type", "general"),
        "summary": row.get("summary", ""),
        "keywords": row.get("keywords", []) if isinstance(row.get("keywords"), list) else [],
        "url": row.get("url", ""),
        "questions": row.get("questions", []) if isinstance(row.get("questions"), list) else [],
        "token_count": row.get("token_count", 0),
    }
    return {
        "page_content": row.get("page_content", "") or "",
        "metadata": meta,
        "doc_key": row.get("doc_key", ""),
    }


def _get_count() -> dict:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks WHERE is_synthetic IS NOT TRUE OR is_synthetic = FALSE")
    real = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks WHERE is_synthetic = TRUE")
    synth = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {"total": total, "real": real, "synth": synth}


def _get_all_real_chunks(limit: int = 10000) -> list[dict]:
    """Fetch all real (non-synthetic) chunks."""
    conn = get_conn()
    cur = conn.cursor()
    from psycopg2.extras import RealDictCursor
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT * FROM chunks
        WHERE is_synthetic IS NOT TRUE OR is_synthetic = FALSE
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

async def main():
    global MAX_VARIANTS_PER_CHUNK, use_noise

    parser = argparse.ArgumentParser(description="Mass synthetic generation via NVIDIA NIM")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL_CHUNKS, help="Target total chunks")
    parser.add_argument("--max-per-source", type=int, default=MAX_VARIANTS_PER_CHUNK, help="Max variants per real chunk")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise injection in query style")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no generation")
    args = parser.parse_args()

    MAX_VARIANTS_PER_CHUNK = args.max_per_source
    use_noise = not args.no_noise
    target = args.target

    ensure_schema()
    counts = _get_count()
    print(f"\n{'='*60}")
    print(f"Current chunks: {counts['total']} total ({counts['real']} real, {counts['synth']} synthetic)")
    print(f"Target: {target}")
    needed = target - counts["total"]
    print(f"Need to generate: {max(0, needed)} synthetic chunks")
    print(f"{'='*60}\n")

    if needed <= 0:
        print(f"Already at {counts['total']} chunks. No generation needed.")
        return

    if args.dry_run:
        print("Dry run — exiting.")
        return

    print(f"Fetching real chunks from DB...")
    real_chunks = _get_all_real_chunks()
    print(f"Loaded {len(real_chunks)} real chunks.")

    filter_model = get_filter_model()

    total_generated = 0
    total_attempts = 0
    counter = 0

    # Shuffle for diversity across sections
    random.shuffle(real_chunks)

    pbar = tqdm(total=needed, desc="Synthetic progress", unit="chunks")

    for row in real_chunks:
        if _get_count()["total"] >= target:
            print(f"\nTarget {target} reached.")
            break

        source = _pg_row_to_source(row)
        source_text = source["page_content"]
        content_type = source["metadata"]["content_type"]

        if len(source_text) < MIN_GEN_CHARS:
            continue

        styles = _pick_styles(content_type)
        source_filter_emb = filter_embed_one(source_text)

        for style in styles:
            if _get_count()["total"] >= target:
                break

            counter += 1
            total_attempts += 1

            prompt = build_prompt(source, style)

            try:
                responses = await call_nim_batch([(prompt, style)])
            except Exception as e:
                print(f"    [NIM error] {e}")
                continue

            generated = responses[0] if responses else ""
            if not generated or len(generated.strip()) < MIN_GEN_CHARS:
                continue

            generated_clean = strip_breadcrumb(generated)
            if len(generated_clean) < MIN_GEN_CHARS:
                generated_clean = generated
            if len(generated_clean) < MIN_GEN_CHARS:
                continue

            if len(generated_clean) > MAX_GEN_CHARS:
                cutoff = generated_clean[:MAX_GEN_CHARS]
                last_period = max(cutoff.rfind('. '), cutoff.rfind('.\n'), cutoff.rfind('! '))
                generated_clean = (
                    generated_clean[:last_period + 1]
                    if last_period > MAX_GEN_CHARS // 2 else cutoff
                )

            # Diversity gate
            gen_filter_emb = filter_embed_one(generated_clean)
            sim_min, sim_max = SIM_THRESHOLDS.get(style, (0.35, 0.88))
            sim_min = max(sim_min, SIM_ABSOLUTE_FLOOR)
            sim = float(np.dot(source_filter_emb, gen_filter_emb))

            if sim < sim_min or sim > sim_max:
                continue

            # Factual guard
            fact_ok, fact_recall = factual_guard(
                source_text, generated_clean, style, content_type,
            )
            if not fact_ok:
                continue

            # Build chunk
            section_title = source["metadata"]["section_title"]
            breadcrumb = f"[{row.get('site','')} | {row.get('title','')}] {section_title} > "
            questions = get_questions(generated_clean, style, source["metadata"])
            qa_pairs = extract_qa_pairs(generated_clean) if style == "qa" else []
            questions_text = " ".join(questions)
            keywords_list = source["metadata"]["keywords"]
            keywords_text = " ".join(keywords_list)

            noise_applied = False
            if use_noise and style == "query" and random.random() < 0.15:
                page_content = inject_noise(source_text)
                noise_applied = True
            elif style == "query":
                page_content = source_text
            else:
                page_content = breadcrumb + generated_clean

            if style == "query":
                enriched_content = " ".join(filter(None, [
                    section_title, keywords_text, questions_text, source_text[:500],
                    source["metadata"]["summary"],
                ]))
            else:
                enriched_content = " ".join(filter(None, [
                    section_title, keywords_text, generated_clean,
                    source["metadata"]["summary"],
                ]))

            if content_type == "study_plan":
                mod_codes = sorted({
                    re.sub(r"\s+", "", m.group(0).upper())
                    for m in re.finditer(r"\b[A-Z]{2}\d[-\s]\d{3}\b|\b[A-Z]{2}\d(?:-\d)?\b", source_text.upper())
                })
                if mod_codes:
                    enriched_content = " ".join(mod_codes[:36]) + " " + enriched_content

            bm25_text = " ".join(filter(None, [section_title, keywords_text, questions_text]))

            synthetic_id = _make_synthetic_id(
                row.get("site", ""), style, generated_clean, counter,
            )

            meta = {
                "site": row.get("site", ""),
                "title": row.get("title", ""),
                "category": row.get("content_type", "general"),
                "url": row.get("url", ""),
                "header": section_title,
                "section_title": section_title,
                "content_type": content_type,
                "summary": source["metadata"]["summary"],
                "keywords": keywords_list,
                "is_synthetic": True,
                "synthetic_style": style,
                "synthetic_id": synthetic_id,
                "questions": questions,
                "qa_pairs": qa_pairs,
                "has_noise": noise_applied,
                "chunk_length": len(page_content),
                "token_count": len(page_content) // 4,
                "filter_sim_score": round(float(sim), 3),
                "similarity_score": round(float(sim), 3),
                "factual_recall": round(float(fact_recall), 3),
            }

            chunk = {
                "page_content": page_content,
                "enriched_content": enriched_content,
                "bm25_text": bm25_text,
                "metadata": meta,
            }

            doc_key = f"{row.get('site','')}|{row.get('title','')}|{section_title}|syn_{style}"

            try:
                ok = insert_chunk(
                    doc_key=doc_key,
                    page_content=page_content,
                    enriched_content=enriched_content,
                    bm25_text=bm25_text,
                    metadata=meta,
                )
                if ok:
                    total_generated += 1
                    pbar.update(1)
            except Exception as e:
                print(f"    [insert error] {e}")

    pbar.close()

    counts = _get_count()
    print(f"\n{'='*60}")
    print(f"Done! Generated {total_generated} synthetic chunks via NVIDIA NIM")
    print(f"Total chunks now: {counts['total']}")
    print(f"Attempts: {total_attempts}, Accepted: {total_generated}, "
          f"Rejected: {total_attempts - total_generated}")
    print(f"{'='*60}\n")

    # ── Post-generation: re-index so new synthetics are searchable ──
    if total_generated > 0:
        print("\nRe-indexing all unindexed chunks...")
        try:
            from rag.pipeline import index_unindexed
            index_unindexed(use_graph=True, gen_synthetic=True, synth_limit=50)
        except Exception as e:
            print(f"Re-index warning: {e}")
        print("Re-index complete.\n")


def run():
    """Entry point for python run_nim_synthetic.py"""
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    run()
