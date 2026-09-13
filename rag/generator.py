"""
rag/generator.py - grounded RAG answers with fallback.
Groundedness thresold is 0.40 based on testing (in-scope ~0.57-0.62, out-of-scope ~0.12-0.18).
"""

from __future__ import annotations

from rag.retriever import retrieve

# emprically measured cutoff
GROUNDEDNESS_THRESHOLD: float = 0.40

FALLBACK_ANSWER = (
    "I don't know — this query appears to be outside my knowledge base. "
    "Please contact Ola customer support directly for further assistance."
)


def generate(
    query: str,
    collection_name: str = "ola_sentence",
    k: int = 3,
) -> dict:
    """fetches chunks and calls mock llm if score >= threshold"""
    chunks = retrieve(query, collection_name, k=k)
    top_score = chunks[0]["score"] if chunks else 0.0

    if top_score < GROUNDEDNESS_THRESHOLD:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "top_score": top_score,
            "grounded": False,
            "chunks": chunks,
        }

    context = "\n\n".join(
        f"[{c['doc_id']} | {c['topic']}]\n{c['text']}" for c in chunks
    )
    sources = list({c["doc_id"] for c in chunks})

    # Import here to avoid circular imports at module load time
    from mock_llm.mock_llm import get_mock_response

    answer = get_mock_response(query=query, context=context)
    return {
        "answer": answer,
        "sources": sources,
        "top_score": top_score,
        "grounded": True,
        "chunks": chunks,
    }


# cached version
def generate_cached(
    query: str,
    collection_name: str = "ola_sentence",
) -> dict:
    """checks in-memory cache first before doing retrieval"""
    from governance.cache import cached_generate  # lazy import to avoid circular dependency

    return cached_generate(query, collection_name)
