"""
query — CLI: Ask a question via the RAG pipeline.

Usage:
    python -m rag.query "Quels sont les frais de scolarité à IIT ?"
    python -m rag.query "Décris le programme de mobilité" --use-graph --top-k 10
"""

import argparse
import json

from rag.pipeline import query
from rag.store import ensure_schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--use-graph", action="store_true", help="Enable Neo4j graph search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    question = args.question or input("Question: ")
    ensure_schema()
    results = query(question, top_k=args.top_k, use_graph=args.use_graph)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n=== Top {len(results)} Results ===")
        for i, r in enumerate(results, 1):
            print(f"\n--- #{i} [{r['content_type']}] {r['site']} | {r['title']} ---")
            print(f"  Section: {r['section_title']}")
            text = (r["enriched_content"] or r["page_content"] or "")[:300]
            print(f"  {text}...")
