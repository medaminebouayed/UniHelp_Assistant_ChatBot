"""
index — CLI: Index unindexed chunks (synthetic gen + embed + KGE + BM25).

Usage:
    python -m rag.index
    python -m rag.index --no-graph       # skip Neo4j KGE
    python -m rag.index --no-synth       # skip synthetic generation
    python -m rag.index --synth-limit 50 # process 50 real chunks at most
"""

import argparse

from rag.pipeline import index_unindexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index unindexed chunks (synth gen + embed + KGE + BM25)")
    parser.add_argument("--no-graph", action="store_true", help="Skip Neo4j LLM entity extraction")
    parser.add_argument("--no-synth", action="store_true", help="Skip synthetic generation")
    parser.add_argument("--synth-limit", type=int, default=100, help="Max real chunks for synthetic gen")
    args = parser.parse_args()
    index_unindexed(
        use_graph=not args.no_graph,
        gen_synthetic=not args.no_synth,
        synth_limit=args.synth_limit,
    )
