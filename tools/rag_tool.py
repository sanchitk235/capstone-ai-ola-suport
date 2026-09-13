"""
tools/rag_tool.py - crewai tool wrapping rag generator for the retrival agent.
arg schema has 'query' which mock llm matches.
"""

from __future__ import annotations

from crewai.tools import tool

from rag.generator import generate

_DEFAULT_COLLECTION = "ola_sentence"


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """searches ola support kb for relevant policy info"""
    result = generate(query, collection_name=_DEFAULT_COLLECTION, k=3)

    if not result["grounded"]:
        return (
            f"No relevant knowledge-base content found for: {query!r}. "
            f"(top similarity score = {result['top_score']:.3f}, "
            f"threshold = 0.40). "
            "Returning fallback: I don't know."
        )

    sources_str = ", ".join(result["sources"]) or "unknown"
    return (
        f"ANSWER: {result['answer']}\n"
        f"SOURCES: {sources_str}\n"
        f"TOP_SCORE: {result['top_score']:.4f}"
    )
