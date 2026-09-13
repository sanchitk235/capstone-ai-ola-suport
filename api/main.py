"""
api/main.py - fastAPI app entrypoint for ola support agent.
startup event builds the chromadb indexes before taking traffic.
"""

from __future__ import annotations

import os

# telemetry off
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import router
from api.websocket import ws_router

app = FastAPI(
    title="Ola Domain Support Agent",
    description="Ola support agent API using mock LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)


@app.on_event("startup")
def startup_event() -> None:
    """builds chromadb collections from knowledge base"""
    from data.knowledge_base import KB_DOCUMENTS
    from rag.indexer import build_indexes

    print("=== Ola Support Agent — Startup ===")
    build_indexes(KB_DOCUMENTS)
    print("=== Indexes ready. Server accepting requests. ===")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> dict:
    from rag.indexer import collection_stats
    return {
        "status": "ok",
        "llm_mode": "MOCK_LLM",
        "collections": collection_stats(),
    }


# ── Entry-point for direct execution ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
