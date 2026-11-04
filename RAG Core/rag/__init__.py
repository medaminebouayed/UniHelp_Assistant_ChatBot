"""
rag — Persistent Hybrid RAG pipeline for IIT/ISB university Q&A.

Architecture:
  ingest → chunk → LLM augment → PostgreSQL (chunks table)
  index  → embed (PGVector) + entities (Neo4j) + BM25 (pickle)
  query  → dense + sparse + graph → RRF → reranker → answer
"""
