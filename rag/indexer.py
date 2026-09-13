"""
rag/indexer.py - creates chromadb collections for the two chunking strats.
Collections: ola_fixed and ola_sentence (cosine similarty).
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings

from rag.chunker import Chunk

# in-memory chroma client singleton
_client: chromadb.Client | None = None


def _get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.Client(Settings(anonymized_telemetry=False))
    return _client


def get_collection(name: str):
    """fetch or make a collection with cosine distance metric"""
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], collection_name: str) -> int:
    """embeds chunks and upserts them into chroma"""
    from rag.embedder import encode

    if not chunks:
        return 0

    collection = get_collection(collection_name)
    texts = [c.text for c in chunks]
    embeddings = encode(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"doc_id": c.doc_id, "topic": c.topic, "strategy": c.strategy}
            for c in chunks
        ],
    )
    print(f"  Indexed {len(chunks):>3} chunks -> '{collection_name}'")
    return len(chunks)


def build_indexes(kb_documents: list[dict]) -> dict[str, int]:
    """chunks all docs both ways and indexes into seperate collections"""
    from rag.chunker import chunk_all_documents

    print("Building ChromaDB indexes...")
    fixed_chunks, sent_chunks = chunk_all_documents(kb_documents)
    counts = {
        "ola_fixed": index_chunks(fixed_chunks, "ola_fixed"),
        "ola_sentence": index_chunks(sent_chunks, "ola_sentence"),
    }
    print("Both collections indexed successfully.\n")
    return counts


def collection_stats() -> dict[str, int]:
    """returns chunk counts per collection"""
    client = _get_client()
    return {col.name: col.count() for col in client.list_collections()}
