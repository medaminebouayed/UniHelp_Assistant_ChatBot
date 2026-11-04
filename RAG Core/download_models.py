import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sentence_transformers

print("[build] Downloading dense model...")
sentence_transformers.SentenceTransformer("intfloat/multilingual-e5-large")
print("[build] Dense model ready.")

print("[build] Downloading reranker (CrossEncoder)...")
sentence_transformers.SentenceTransformer("BAAI/bge-reranker-v2-m3")
print("[build] Reranker ready.")

print("[build] Downloading filter model (MiniLM)...")
sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
print("[build] Filter model ready.")
