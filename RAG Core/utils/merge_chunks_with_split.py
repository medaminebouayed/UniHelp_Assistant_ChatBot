"""
merge_chunks_with_split.py — Merge then split long chunks at sub-chunk boundaries.
===============================================================================
Problem:
  - Merged corpus has 1 chunk per doc_key, but long mobility chunks (up to 44K chars)
    get truncated to 2048 in the cross-encoder, hiding relevant content.
  - The original merge concatenated sub-chunks with "\n\n".
  - Splitting at "\n\n" boundaries keeps each sub-chunk as a focused unit ≤ 2048 chars.

Output:
  - corpus_merged_split.json — each split chunk gets a unique sub_key:
      "parent_doc_key||part_N"
  - Metadata and questions are preserved on every split.
  - Chunks ≤ 2048 chars are kept as-is (single part).

Usage:
    python utils/merge_chunks_with_split.py \
        --input data/synthetic/corpus_final.json \
        --output data/synthetic/corpus_merged_split.json
"""

import json
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple

MAX_CHARS = 2048

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def doc_key(chunk: Dict) -> str:
    m = chunk.get("metadata", {})
    return (
        m.get("site", "").strip().lower() + "|"
        + m.get("title", "").strip().lower() + "|"
        + (m.get("section_title") or m.get("header") or "").strip().lower()
    )


def split_text_at_boundaries(text: str, max_chars: int) -> List[str]:
    """Split text at \\n\\n boundaries, keeping parts ≤ max_chars."""
    paragraphs = text.split("\n\n")
    parts = []
    current = []
    current_len = 0
    for para in paragraphs:
        if not para.strip():
            continue
        para_len = len(para)
        if current_len + para_len + (2 if current else 0) <= max_chars:
            current.append(para)
            current_len += para_len + (2 if current else 0)
        else:
            if current:
                parts.append("\n\n".join(current))
            if para_len > max_chars:
                single_parts = split_long_paragraph(para, max_chars)
                parts.extend(single_parts)
                current = []
                current_len = 0
            else:
                current = [para]
                current_len = para_len
    if current:
        parts.append("\n\n".join(current))
    return parts


def split_long_paragraph(para: str, max_chars: int) -> List[str]:
    """Split a single long paragraph (no \\n\\n) at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.?!])\s+', para)
    parts = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + (1 if current else 0) <= max_chars:
            current.append(sent)
            current_len += sent_len + (1 if current else 0)
        else:
            if current:
                parts.append(" ".join(current))
            if sent_len > max_chars:
                # Last resort: hard split at max_chars
                for i in range(0, sent_len, max_chars):
                    parts.append(sent[i:i + max_chars])
                current = []
                current_len = 0
            else:
                current = [sent]
                current_len = sent_len
    if current:
        parts.append(" ".join(current))
    return parts


def merge_and_split_group(chunks: List[Dict]) -> List[Dict]:
    """Merge chunks sharing doc_key, then split if > MAX_CHARS."""
    first = chunks[0]
    meta = dict(first.get("metadata", {}))

    page_contents = []
    enriched_contents = []
    bm25_texts = []
    seen_q = set()
    all_questions = []
    seen_k = set()
    all_keywords = []
    all_qa = []
    total_length = 0
    total_tokens = 0

    for c in chunks:
        pc = c.get("page_content", "") or ""
        if pc:
            page_contents.append(pc)
        ec = c.get("enriched_content", "") or ""
        if ec:
            enriched_contents.append(ec)
        bm = c.get("bm25_text", "") or ""
        if bm:
            bm25_texts.append(bm)

        cm = c.get("metadata", {})
        for q in (cm.get("questions") or []):
            if q and q.strip() not in seen_q:
                seen_q.add(q.strip())
                all_questions.append(q.strip())
        for k in (cm.get("keywords") or []):
            if k and k.strip() not in seen_k:
                seen_k.add(k.strip())
                all_keywords.append(k.strip())
        for qa in (cm.get("qa_pairs") or []):
            if qa not in all_qa:
                all_qa.append(qa)

        total_length += cm.get("chunk_length", 0) or len(pc)
        total_tokens += cm.get("token_count", 0) or 0

    merged_text = "\n\n".join(page_contents)
    merged_enriched = "\n\n".join(enriched_contents)
    merged_bm25 = "\n\n".join(bm25_texts)

    enriched_parts = split_text_at_boundaries(merged_enriched, MAX_CHARS)

    base_key = doc_key(first)
    parent_section = meta.get("section_title") or meta.get("header") or ""

    result = []
    for part_idx, enriched_part in enumerate(enriched_parts):
        sub_key = base_key + "||part_" + str(part_idx)

        # Split page_content and bm25_text proportionally (rough alignment)
        page_part = _get_proportional_part(merged_text, merged_enriched, enriched_part)
        bm25_part = _get_proportional_part(merged_bm25, merged_enriched, enriched_part)

        part_meta = dict(meta)
        part_meta["questions"] = list(all_questions)
        part_meta["keywords"] = list(all_keywords)
        part_meta["qa_pairs"] = list(all_qa)
        part_meta["chunk_length"] = len(page_part)
        part_meta["token_count"] = total_tokens
        part_meta["is_synthetic"] = False
        part_meta["synthetic_style"] = None
        part_meta["synthetic_id"] = None
        part_meta["has_noise"] = False
        part_meta["sub_index"] = part_idx
        part_meta["section_title"] = parent_section

        result.append(OrderedDict([
            ("page_content", page_part),
            ("enriched_content", enriched_part),
            ("bm25_text", bm25_part or enriched_part),
            ("metadata", part_meta),
        ]))
    return result


def _get_proportional_part(full_text: str, full_enriched: str, enriched_part: str) -> str:
    """Get the proportionally corresponding part of full_text based on enriched_part position."""
    if not full_enriched or not full_text:
        return enriched_part
    pos = full_enriched.find(enriched_part)
    if pos < 0:
        return enriched_part
    ratio = pos / max(len(full_enriched), 1)
    text_pos = int(ratio * len(full_text))
    part_ratio = len(enriched_part) / max(len(full_enriched), 1)
    text_len = int(part_ratio * len(full_text))
    return full_text[text_pos:text_pos + text_len]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge + split chunks per doc_key")
    parser.add_argument("--input", required=True, help="Input corpus JSON")
    parser.add_argument("--output", required=True, help="Output corpus JSON")
    parser.add_argument("--max-chars", type=int, default=2048,
                        help="Max chars per split chunk (default 2048)")
    args = parser.parse_args()

    global MAX_CHARS
    MAX_CHARS = args.max_chars

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"Loading corpus from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Corpus: {len(corpus)} chunks")

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in corpus:
        groups[doc_key(c)].append(c)

    print(f"Unique doc_keys: {len(groups)}")
    multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"Doc_keys with multiple chunks: {multi}")

    merged = []
    total_parts = 0
    split_count = 0
    for dk, chunks in groups.items():
        if len(chunks) == 1:
            merged.append(chunks[0])
            total_parts += 1
        else:
            parts = merge_and_split_group(chunks)
            merged.extend(parts)
            total_parts += len(parts)
            if len(parts) > 1:
                split_count += 1

    print(f"Split {split_count} doc_keys into multiple parts")
    print(f"Total output chunks: {total_parts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Written to {output_path}")

    # Distribution stats
    from collections import Counter
    type_counts = Counter()
    for c in merged:
        type_counts[c.get("metadata", {}).get("content_type", "general")] += 1
    print(f"Content types: {dict(type_counts)}")

    sub_counts = Counter()
    for c in merged:
        ec = c.get("enriched_content", "")
        sub_counts["over_2048"] += 1 if len(ec) > MAX_CHARS else 0
    print(f"Chunks > {MAX_CHARS} chars: {sub_counts['over_2048']}/{total_parts}")


if __name__ == "__main__":
    main()
