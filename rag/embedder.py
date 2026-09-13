"""
rag/embedder.py - sentence transformers wrapper for all-MiniLM-L6-v2.
generates 384 dim vectors locally, no key needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# lazy load singleton so import stays quick
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """batch encode strings to normalized vectors for chromadb"""
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit normalize so dot product = cosine
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.tolist()


def encode_single(text: str) -> list[float]:
    """quick helper for single string embeding"""
    return encode([text])[0]
