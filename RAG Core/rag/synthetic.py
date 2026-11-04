"""
synthetic — Incremental synthetic chunk generation for new real chunks.

Generates 2-4 synthetic variants per new real chunk using diverse styles,
reusing prompts and quality gates from synthetic_chunks.py.

Usage:
    from rag.synthetic import generate_for_chunks
    chunks = generate_for_chunks(pg_rows)
"""

import re
import json
import hashlib
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.store import insert_chunk, get_real_chunks_without_synthetic, mark_synthetic_generated, ensure_schema
from rag.config import PROJECT_ROOT

from utils.augmentation.synthetic_chunks import (
    STYLES, STYLE_WEIGHTS, STUDY_PLAN_STYLE_WEIGHTS,
    SIM_THRESHOLDS, SIM_ABSOLUTE_FLOOR,
    MAX_GEN_CHARS, MIN_GEN_CHARS,
    build_prompt, factual_guard,
    get_questions, extract_qa_pairs,
    strip_breadcrumb, inject_noise,
    call_ollama_batch,
    filter_embed_one, get_filter_model,
)


# ── Config ──────────────────────────────────────────────────────────────

VARIANTS_PER_CHUNK = int(3)  # how many synthetic variants per real chunk
MAX_VARIANTS_PER_CHUNK = int(4)
NOISE_RATE = 0.08

STYLE_CHOICES = list(STYLE_WEIGHTS.keys())


def _pg_row_to_source(row: dict) -> dict:
    """Convert a PostgreSQL row to the source format expected by synthetic_chunks functions."""
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


def _pick_styles(content_type: str) -> List[str]:
    """Pick 2-4 styles for a chunk based on content type."""
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


def generate_for_chunks(
    pg_rows: List[dict],
    use_noise: bool = True,
    batch_callback=None,
) -> int:
    """Generate synthetic variants for a list of real chunks (PG row format).

    Each real chunk gets 2-4 synthetic variants in diverse styles.
    Quality gates: similarity to source, factual recall, length checks.
    Marks source chunks as synth_generated when done.

    Args:
        pg_rows: List of dicts from store (real chunks without synthetic variants).
        use_noise: Inject typos in query-style variants.
        batch_callback: Optional callback(processed, total).

    Returns:
        Number of synthetic chunks successfully generated and inserted.
    """
    if not pg_rows:
        return 0

    real_chunks = [r for r in pg_rows if not r.get("is_synthetic")]
    if not real_chunks:
        print("[synthetic] No real chunks to generate from.")
        return 0

    print(f"[synthetic] Generating variants for {len(real_chunks)} real chunks...")

    filter_model = get_filter_model()

    total_generated = 0
    total_attempts = 0

    counter = 0

    for row_idx, row in enumerate(tqdm(real_chunks, desc="Synthetic")):
        source = _pg_row_to_source(row)
        source_text = source["page_content"]
        content_type = source["metadata"]["content_type"]

        if len(source_text) < MIN_GEN_CHARS:
            continue

        styles = _pick_styles(content_type)
        source_filter_emb = filter_embed_one(source_text)

        for style in styles:
            counter += 1
            total_attempts += 1

            prompt = build_prompt(source, style)

            try:
                responses = call_ollama_batch([(prompt, style)])
            except Exception as e:
                print(f"    [Ollama error] {e}")
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

            # Diversity gate: similarity to source
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

            # Build chunk dict
            breadcrumb = f"[{row.get('site','')} | {row.get('title','')}] {row.get('section_title','')} > "
            questions = get_questions(generated_clean, style, source["metadata"])
            qa_pairs = extract_qa_pairs(generated_clean) if style == "qa" else []
            questions_text = " ".join(questions)
            keywords_list = source["metadata"]["keywords"]
            keywords_text = " ".join(keywords_list)
            section_title = source["metadata"]["section_title"]

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
                row.get("site", ""), style,
                generated_clean, counter,
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
            except Exception as e:
                print(f"    [insert error] {e}")

            if batch_callback:
                batch_callback(total_generated, total_attempts)

    print(f"[synthetic] Generated {total_generated} variants from {len(real_chunks)} real chunks "
          f"({total_attempts} attempts, {total_attempts - total_generated} rejected).")

    # Mark source chunks as synthetic-generated so they're not re-processed
    source_doc_keys = [r["doc_key"] for r in real_chunks if r.get("doc_key")]
    if source_doc_keys:
        mark_synthetic_generated(source_doc_keys)

    return total_generated


def generate_for_unindexed(
    use_noise: bool = True,
    limit: int = 100,
) -> int:
    """Convenience: fetch real chunks without synthetic variants → generate → insert.

    Args:
        use_noise: Inject typos in query-style variants.
        limit: Max real chunks to process.

    Returns:
        Number of synthetic chunks inserted.
    """
    ensure_schema()
    chunks = get_real_chunks_without_synthetic(limit=limit)
    if not chunks:
        print("[synthetic] No real chunks needing synthetic generation.")
        return 0
    return generate_for_chunks(chunks, use_noise=use_noise)
