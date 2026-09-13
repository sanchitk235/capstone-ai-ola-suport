"""
crew/schemas.py - pydantic output schema for crew reponses.
validates answer structure before sending back.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CrewResponse(BaseModel):
    """output format enforced on every crew run"""

    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="The composed, grounded answer text")
    sources: list[str] = Field(
        default_factory=list,
        description="KB document IDs cited in the answer (e.g. ['KB-004'])",
    )
    ticket_status: Optional[str] = Field(
        None,
        description="Status of the looked-up ticket, if applicable",
    )
    escalation_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Computed escalation score in [0, 1], if a ticket was looked up",
    )
    recommend_escalation: Optional[bool] = Field(
        None,
        description="Whether escalation is recommended based on the score",
    )
    grounded: bool = Field(
        ...,
        description="True if the answer is supported by retrieved KB context",
    )
    session_id: str = Field(..., description="Conversation session identifier")
    trace_id: str = Field(..., description="Request trace ID for logging")
