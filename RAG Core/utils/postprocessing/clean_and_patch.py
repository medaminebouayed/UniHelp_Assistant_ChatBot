"""
clean_and_patch.py — Post-generation cleaner + original input patcher
======================================================================
Location : utils/augmentation/clean_and_patch.py
Run from : project root

Three commands:

    # Clean existing synthetic chunks (all 5 fixes + IMP 1 qa_pairs recovery)
    py utils/augmentation/clean_and_patch.py clean \
        --input  data/synthetic/synthetic_chunks.json \
        --output data/synthetic/synthetic_chunks_clean.json

    # Patch original/augmented chunks with missing fields
    py utils/augmentation/clean_and_patch.py patch \
        --input  data/augmented/iit_chunks_augmented.json \
        --output data/augmented/iit_chunks_patched.json

    # Clean synthetic + patch originals + merge into one corpus
    py utils/augmentation/clean_and_patch.py all \
        --synthetic data/synthetic/synthetic_chunks.json \
        --originals data/augmented/iit_chunks_augmented.json \
                    data/augmented/isb_chunks_augmented.json \
        --output    data/synthetic/full_corpus.json

Changes from previous version:
  - clean_synthetic() now also rebuilds qa_pairs from existing questions
    when qa_pairs is missing (recovery for old synthetic chunks)
  - patch_originals() adds qa_pairs = [] field for schema consistency
  - Both functions add has_noise = False for schema consistency
  - IMP 2: per-style sim threshold documented in clean report
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Optional

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

_BREADCRUMB_RE   = re.compile(r'^\[.*?\|.*?\].*?>\s*', re.DOTALL)
_MULTI_BREADCRUMB = re.compile(r'(\[.*?\|.*?\].*?>)\s*(\[.*?\|.*?\].*?>)', re.DOTALL)

MAX_QUESTION_LEN = 120
MAX_ANSWER_LEN   = 400


# ════════════════════════════════════════════════════════════════════════
# QA PAIR RECOVERY (for old synthetic chunks without qa_pairs)
# ════════════════════════════════════════════════════════════════════════

def recover_qa_pairs_from_text(text: str) -> list[dict]:
    """
    Attempt to extract Q/R pairs from raw page_content of a QA-style chunk.
    Used when qa_pairs field is missing from old synthetic data.
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
            while j < len(lines):
                nxt = lines[j]
                if nxt.startswith("Q:"):
                    break
                if nxt.startswith("R:"):
                    answer_parts.append(nxt[2:].strip())
                elif answer_parts:
                    answer_parts.append(nxt)
                j += 1
            a = " ".join(answer_parts).strip()[:MAX_ANSWER_LEN]
            if q:
                pairs.append({"q": q, "a": a})
            i = j
        else:
            i += 1

    return pairs


# ════════════════════════════════════════════════════════════════════════
# CLEAN: fix all 5 issues + IMP 1 qa_pairs recovery
# ════════════════════════════════════════════════════════════════════════

def clean_synthetic(chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    Apply all fixes and improvements to existing synthetic chunks.
    Safe to run multiple times (idempotent).
    """
    stats = {
        "total"             : len(chunks),
        "fixed_breadcrumb"  : 0,
        "fixed_enriched"    : 0,
        "fixed_questions"   : 0,
        "fixed_query_pc"    : 0,
        "removed_long_qs"   : 0,
        "removed_empty_qs"  : 0,
        "recovered_qa_pairs": 0,
        "added_noise_flag"  : 0,
    }
    cleaned: list[dict] = []

    for c in chunks:
        c    = dict(c)
        meta = dict(c.get("metadata", {}))
        c["metadata"] = meta

        style         = meta.get("synthetic_style", "")
        pc            = c.get("page_content", "")
        ec            = c.get("enriched_content", "")
        qs            = meta.get("questions", [])
        qa_pairs      = meta.get("qa_pairs", [])
        section_title = meta.get("section_title", meta.get("header", ""))
        summary       = meta.get("summary", "")
        keywords_text = " ".join(meta.get("keywords", []))

        # ── FIX 3: remove duplicate breadcrumb ─────────────────────────
        if _MULTI_BREADCRUMB.search(pc):
            pc = _MULTI_BREADCRUMB.sub(r'\1 ', pc).strip()
            c["page_content"] = pc
            stats["fixed_breadcrumb"] += 1

        # ── FIX 5: query style — page_content should not be only bullets ─
        if style == "query":
            text_after = _BREADCRUMB_RE.sub('', pc).strip()
            all_bullets = text_after and all(
                line.strip().startswith(("-", "•"))
                for line in text_after.splitlines()
                if line.strip()
            )
            if all_bullets:
                # Rescue bullet queries into questions list
                bullet_qs = []
                for line in text_after.splitlines():
                    q = line.strip().lstrip("-•").strip()
                    if len(q) > 5:
                        bullet_qs.append(q[:MAX_QUESTION_LEN])
                if bullet_qs:
                    qs = bullet_qs
                    meta["questions"] = qs
                # Replace page_content with a meaningful placeholder
                placeholder = " ".join(filter(None, [section_title, summary]))
                if placeholder:
                    c["page_content"] = placeholder
                    pc = placeholder
                stats["fixed_query_pc"] += 1

        # ── FIX 2: clean questions — remove empty, cap length ───────────
        cleaned_qs: list[str] = []
        for q in qs:
            q = q.strip()
            if not q:
                stats["removed_empty_qs"] += 1
                continue
            if len(q) > MAX_QUESTION_LEN:
                stats["removed_long_qs"] += 1
                q = q[:MAX_QUESTION_LEN].strip()
                if len(q) < 10:
                    continue
            cleaned_qs.append(q)
        meta["questions"] = cleaned_qs
        if len(cleaned_qs) != len(qs):
            stats["fixed_questions"] += 1

        # ── IMP 1: recover qa_pairs if missing for QA-style chunks ──────
        if style == "qa" and not qa_pairs:
            # Try to recover from page_content
            recovered = recover_qa_pairs_from_text(pc)
            if recovered:
                meta["qa_pairs"] = recovered
                qa_pairs = recovered
                stats["recovered_qa_pairs"] += 1
            else:
                meta["qa_pairs"] = []
        elif "qa_pairs" not in meta:
            meta["qa_pairs"] = []

        # ── Ensure has_noise field exists ───────────────────────────────
        if "has_noise" not in meta:
            meta["has_noise"] = False
            stats["added_noise_flag"] += 1

        # ── FIX 4: rebuild enriched_content if it equals page_content ───
        questions_text  = " ".join(cleaned_qs)
        qa_answers_text = " ".join(p["a"] for p in qa_pairs if p.get("a"))
        generated_text  = _BREADCRUMB_RE.sub('', pc).strip()

        if not ec or ec.strip() == pc.strip():
            c["enriched_content"] = " ".join(filter(None, [
                section_title,
                summary,
                questions_text,
                qa_answers_text,
                keywords_text,
                generated_text,
            ]))
            stats["fixed_enriched"] += 1

        # Rebuild bm25_text for consistency
        c["bm25_text"] = " ".join(filter(None, [
            section_title,
            summary,
            keywords_text,
            generated_text,
            questions_text,
            qa_answers_text,
        ]))

        cleaned.append(c)

    return cleaned, stats


# ════════════════════════════════════════════════════════════════════════
# PATCH: add missing fields to original/augmented chunks
# ════════════════════════════════════════════════════════════════════════

def patch_originals(chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    Add all fields that synthetic chunks have so the corpus schema is uniform.
    Safe to run multiple times (idempotent — only sets missing fields).

    Fields added if missing:
      top-level : enriched_content, bm25_text
      metadata  : is_synthetic, synthetic_style, synthetic_id, similarity_score,
                  questions, qa_pairs, has_noise, section_title, content_type,
                  summary, keywords
    """
    stats = {
        "total"             : len(chunks),
        "patched_enriched"  : 0,
        "patched_bm25"      : 0,
        "patched_flags"     : 0,
        "patched_qa_pairs"  : 0,
        "patched_section_title": 0,
    }
    patched: list[dict] = []

    for c in chunks:
        c    = dict(c)
        meta = dict(c.get("metadata", {}))
        c["metadata"] = meta

        pc            = c.get("page_content", "")
        section_title = meta.get("section_title", meta.get("header", ""))
        summary       = meta.get("summary", "")
        keywords      = meta.get("keywords", [])
        questions     = meta.get("questions", [])
        keywords_text = " ".join(keywords)
        questions_text= " ".join(questions)

        # ── Synthetic identity flags ────────────────────────────────────
        if "is_synthetic" not in meta:
            meta["is_synthetic"]     = False
            meta["synthetic_style"]  = None
            meta["synthetic_id"]     = None
            meta["similarity_score"] = None
            meta["has_noise"]        = False
            stats["patched_flags"] += 1

        # ── Section title ───────────────────────────────────────────────
        if "section_title" not in meta:
            meta["section_title"] = meta.get("header", "")
            stats["patched_section_title"] += 1

        # ── Content type ────────────────────────────────────────────────
        if "content_type" not in meta:
            meta["content_type"] = "general"

        # ── Summary and keywords (present if augmented, else empty) ─────
        if "summary" not in meta:
            meta["summary"] = ""
        if "keywords" not in meta:
            meta["keywords"] = []

        # ── IMP 1: qa_pairs ─────────────────────────────────────────────
        if "qa_pairs" not in meta:
            meta["qa_pairs"] = []
            stats["patched_qa_pairs"] += 1

        # ── questions (flat list) ────────────────────────────────────────
        if "questions" not in meta:
            meta["questions"] = []

        # Refresh text vars after possible meta updates
        section_title = meta.get("section_title", "")
        summary       = meta.get("summary", "")
        keywords_text = " ".join(meta.get("keywords", []))
        questions_text= " ".join(meta.get("questions", []))
        qa_answers    = " ".join(
            p["a"] for p in meta.get("qa_pairs", []) if p.get("a")
        )

        # ── enriched_content ────────────────────────────────────────────
        if not c.get("enriched_content"):
            c["enriched_content"] = " ".join(filter(None, [
                section_title,
                summary,
                questions_text,
                qa_answers,
                keywords_text,
                pc,
            ]))
            stats["patched_enriched"] += 1

        # ── bm25_text ────────────────────────────────────────────────────
        if not c.get("bm25_text"):
            c["bm25_text"] = " ".join(filter(None, [
                section_title,
                summary,
                keywords_text,
            ]))
            stats["patched_bm25"] += 1

        patched.append(c)

    return patched, stats


# ════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════

def print_report(stats: dict, label: str = ""):
    print(f"\n── Report {label} ─────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")
    print(f"────────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════
# PATH HELPER
# ════════════════════════════════════════════════════════════════════════

def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="command")

    # clean
    p_clean = sub.add_parser("clean", help="Clean existing synthetic chunks")
    p_clean.add_argument("--input",  required=True)
    p_clean.add_argument("--output", required=True)

    # patch
    p_patch = sub.add_parser("patch", help="Patch original chunks with missing fields")
    p_patch.add_argument("--input",  required=True)
    p_patch.add_argument("--output", required=True)

    # all
    p_all = sub.add_parser("all", help="Clean synthetic + patch originals + merge")
    p_all.add_argument("--synthetic", required=True)
    p_all.add_argument(
        "--originals", nargs="+",
        default=[
            str(DATA_DIR / "augmented" / "iit_chunks_augmented.json"),
            str(DATA_DIR / "augmented" / "isb_chunks_augmented.json"),
        ],
    )
    p_all.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "clean":
        inp, out = resolve(args.input), resolve(args.output)
        with open(inp, encoding="utf-8") as f:
            chunks = json.load(f)
        cleaned, stats = clean_synthetic(chunks)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print_report(stats, inp.name)
        print(f"Saved {len(cleaned)} chunks → {out}")

    elif args.command == "patch":
        inp, out = resolve(args.input), resolve(args.output)
        with open(inp, encoding="utf-8") as f:
            chunks = json.load(f)
        patched, stats = patch_originals(chunks)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(patched, f, ensure_ascii=False, indent=2)
        print_report(stats, inp.name)
        print(f"Saved {len(patched)} chunks → {out}")

    elif args.command == "all":
        # 1. Clean synthetic
        syn_path = resolve(args.synthetic)
        with open(syn_path, encoding="utf-8") as f:
            syn_chunks = json.load(f)
        cleaned_syn, syn_stats = clean_synthetic(syn_chunks)
        print_report(syn_stats, f"synthetic ({syn_path.name})")

        # 2. Patch originals
        all_real: list[dict] = []
        for orig_str in args.originals:
            p = resolve(orig_str)
            if not p.exists():
                print(f"  [warn] {p} not found, skipping.")
                continue
            with open(p, encoding="utf-8") as f:
                orig_chunks = json.load(f)
            patched, patch_stats = patch_originals(orig_chunks)
            print_report(patch_stats, p.name)
            all_real.extend(patched)

        # 3. Merge and save
        merged = all_real + cleaned_syn
        out    = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\nMerged: {len(all_real)} real + {len(cleaned_syn)} synthetic = {len(merged)} total")
        print(f"Saved → {out}")

    else:
        parser.print_help()
        sys.exit(1)