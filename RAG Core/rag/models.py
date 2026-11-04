"""
models — Lazy-loaded model singletons (GPU-accelerated reranker).

- Dense model stays on CPU (good enough with reduced TOP_K_DENSE=40)
- Reranker uses CUDA GPU at full precision (biggest speedup: 8-15s → <1s)
"""

from functools import lru_cache
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.config import DENSE_MODEL_NAME, RERANKER_NAME

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@lru_cache(maxsize=1)
def get_dense_model() -> SentenceTransformer:
    # FORCE CPU: Prevents VRAM exhaustion when Ollama is running
    device = 'cpu'
    print(f"[models] Loading dense model: {DENSE_MODEL_NAME} on {device}")
    model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    return model


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    print(f"[models] Loading reranker: {RERANKER_NAME} on {_DEVICE}")
    model = CrossEncoder(RERANKER_NAME, device=_DEVICE)
    if _DEVICE == 'cuda':
        model.model.half()  # Halve VRAM usage (saves ~1.1GB)
    return model
