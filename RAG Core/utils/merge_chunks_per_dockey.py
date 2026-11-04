"""
merge_chunks_per_dockey.py — Merge all chunks sharing the same doc_key into one.
===============================================================================
Why: Mobility chunks have up to 31 chunks per doc_key, each covering a different
subtopic (diplomas, culture, costs, etc.). Dense dedup keeps only the BEST-scoring
chunk, discarding the other 30. Merging gives each doc_key ONE rich embedding
that captures ALL subtopics.

Usage:
    python utils/merge_chunks_per_dockey.py \\
        --input data/synthetic/corpus_final.json \\
        --output data/synthetic/corpus_merged.json
"""

import json
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def doc_key(chunk: Dict) -> str:
    m = chunk.get("metadata", {})
    return (
        m.get("site", "").strip().lower() + "|"
        + m.get("title", "").strip().lower() + "|"
        + (m.get("section_title") or m.get("header") or "").strip().lower()
    )


def merge_group(chunks: List[Dict]) -> Dict:
    """Merge a list of chunks with the same doc_key into a single chunk."""
    first = chunks[0]
    meta = dict(first.get("metadata", {}))

    # Concatenate text fields
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

    # Update metadata
    meta["questions"] = all_questions
    meta["keywords"] = all_keywords
    meta["qa_pairs"] = all_qa
    meta["chunk_length"] = len(merged_text)
    meta["token_count"] = total_tokens
    meta["is_synthetic"] = False
    meta["synthetic_style"] = None
    meta["synthetic_id"] = None
    meta["has_noise"] = False

    return OrderedDict([
        ("page_content", merged_text),
        ("enriched_content", merged_enriched or merged_text),
        ("bm25_text", merged_bm25 or merged_text),
        ("metadata", meta),
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge chunks per doc_key")
    parser.add_argument("--input", required=True, help="Input corpus JSON")
    parser.add_argument("--output", required=True, help="Output merged corpus JSON")
    args = parser.parse_args()

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

    # Group by doc_key
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in corpus:
        groups[doc_key(c)].append(c)

    print(f"Unique doc_keys: {len(groups)}")
    multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"Doc_keys with multiple chunks: {multi}")

    # Merge each group
    merged = []
    for dk, chunks in groups.items():
        if len(chunks) == 1:
            merged.append(chunks[0])
        else:
            merged.append(merge_group(chunks))

    print(f"Merged corpus: {len(merged)} chunks")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
