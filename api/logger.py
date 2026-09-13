"""
api/logger.py - elk style json lines logger.
prints structured json entries to stdout with masked pii.
"""

from __future__ import annotations

import json
import sys
import time
import uuid


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_request(
    *,
    query: str,
    response: str,
    session_id: str,
    trace_id: str,
    duration_ms: float,
    grounded: bool = True,
    cache_hit: bool = False,
    extra: dict | None = None,
) -> None:
    """prints single json line with request metrics and masked query"""
    entry: dict = {
        "trace_id": trace_id,
        "timestamp": _now_iso(),
        "session_id": session_id,
        "query": query,           # caller guarantees this is already masked
        "response_excerpt": response[:200],
        "duration_ms": round(duration_ms, 2),
        "grounded": grounded,
        "cache_hit": cache_hit,
    }
    if extra:
        entry.update(extra)

    print(json.dumps(entry, ensure_ascii=False), file=sys.stdout, flush=True)
