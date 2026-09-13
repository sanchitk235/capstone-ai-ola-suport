"""
rag/retriever.py - top k retrival from chromadb collections.
score = 1 - distance for cosine metric.
"""

from __future__ import annotations

from rag.embedder import encode_single
from rag.indexer import get_collection


def retrieve(
    query: str,
    collection_name: str,
    k: int = 3,
) -> list[dict]:
    """finds top k chunks for query ordered by cosine score"""
    collection = get_collection(collection_name)
    total = collection.count()
    if total == 0:
        return []

    q_emb = encode_single(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    items: list[dict] = []
    for doc_text, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        score = max(0.0, 1.0 - dist)  # convert cosine distance → similarity
        items.append(
            {
                "text": doc_text,
                "doc_id": meta["doc_id"],
                "topic": meta["topic"],
                "chunk_id": chunk_id,
                "score": round(score, 4),
            }
        )

    return sorted(items, key=lambda x: x["score"], reverse=True)
