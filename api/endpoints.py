"""
api/endpoints.py - fastAPI routes for /ask and /add-document.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from crew.guardrails import apply_input_guardrails, PromptInjectionError, mask_pii
from crew.crew import run_crew
from governance.risk_classification import BudgetExceededError
from api.logger import log_request

router = APIRouter()


# request/response schemas
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="User's question")
    session_id: str = Field("default", description="Conversation session identifier")
    record_id: Optional[str] = Field(None, description="Optional TKT-NNN ticket to look up")


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    sources: list[str]
    grounded: bool
    ticket_status: Optional[str] = None
    escalation_score: Optional[float] = None
    recommend_escalation: Optional[bool] = None
    cache_hit: bool = False


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document ID (e.g. KB-013)")
    topic: str = Field(..., description="Topic slug (e.g. new_policy)")
    content: str = Field(..., min_length=20, description="Document body text")


class AddDocumentResponse(BaseModel):
    success: bool
    chunks_indexed: int
    collection: str


@router.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(req: AskRequest) -> AskResponse:
    """runs crewai pipeline with guardrails and budget check"""
    t0 = time.perf_counter()
    trace_id = str(uuid.uuid4())

    try:
        crew_response = run_crew(
            query=req.query,
            session_id=req.session_id,
            record_id=req.record_id,
        )
    except PromptInjectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request blocked: {exc}",
        )
    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )

    duration_ms = (time.perf_counter() - t0) * 1000

    # Log with pre-masked query (mask_pii applied inside run_crew, but we
    # also mask here defensively before writing to the log)
    log_request(
        query=mask_pii(req.query),
        response=crew_response.answer,
        session_id=req.session_id,
        trace_id=trace_id,
        duration_ms=duration_ms,
        grounded=crew_response.grounded,
    )

    return AskResponse(
        trace_id=trace_id,
        session_id=crew_response.session_id,
        answer=crew_response.answer,
        sources=crew_response.sources,
        grounded=crew_response.grounded,
        ticket_status=crew_response.ticket_status,
        escalation_score=crew_response.escalation_score,
        recommend_escalation=crew_response.recommend_escalation,
    )


# ── POST /add-document ────────────────────────────────────────────────────────

@router.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    """chunks and indexes a new document into chromadb at runtime"""
    from rag.chunker import sentence_chunks
    from rag.indexer import index_chunks

    doc = {"doc_id": req.doc_id, "topic": req.topic, "content": req.content}
    chunks = sentence_chunks(doc)
    count = index_chunks(chunks, "ola_sentence")
    return AddDocumentResponse(
        success=True,
        chunks_indexed=count,
        collection="ola_sentence",
    )
