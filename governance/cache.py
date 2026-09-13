"""
governance/cache.py - in memory query cache.
tracks hit rate and llm call counts to prove cost savings.
"""

from __future__ import annotations

# simple dict cache + stats
_cache: dict[str, dict] = {}
_stats = {"llm_calls": 0, "cache_hits": 0}


def _normalize(query: str) -> str:
    """cleans query to standard lowercase key"""
    return " ".join(query.lower().strip().split())


def cached_generate(
    query: str,
    collection_name: str = "ola_sentence",
) -> dict:
    """returns cached answer or calls generator on cache miss"""
    from rag.generator import generate  # lazy to avoid circular import

    key = _normalize(query)

    if key in _cache:
        _stats["cache_hits"] += 1
        cached = _cache[key].copy()
        cached["cache_hit"] = True
        cached["llm_call_count"] = _stats["llm_calls"]
        return cached

    # Cache miss — call the real generator
    _stats["llm_calls"] += 1
    result = generate(query, collection_name=collection_name)
    _cache[key] = result.copy()

    enriched = result.copy()
    enriched["cache_hit"] = False
    enriched["llm_call_count"] = _stats["llm_calls"]
    return enriched


def cache_stats() -> dict:
    """returns current hit/miss statistics"""
    return {
        "cache_size": len(_cache),
        "llm_calls": _stats["llm_calls"],
        "cache_hits": _stats["cache_hits"],
    }


def clear_cache() -> None:
    """resets cache and call stats"""
    _cache.clear()
    _stats["llm_calls"] = 0
    _stats["cache_hits"] = 0
