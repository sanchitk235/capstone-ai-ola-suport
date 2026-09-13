"""
crew/crew.py - ties together agents, tasks, guardrails, and validation.
executes the full pipeline and stores turn in session memory.
"""

from __future__ import annotations

import os
import re
import uuid

# disable crewai telemetry
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Crew, Process

from crew.agents import retrieval_agent, lookup_agent, composer_agent
from crew.tasks import task_retrieve, task_lookup, task_compose
from crew.guardrails import apply_input_guardrails, PromptInjectionError
from crew.memory import add_turn
from crew.schemas import CrewResponse
from governance.risk_classification import budget_check, BudgetExceededError

# sequential crew definition
_crew = Crew(
    agents=[retrieval_agent, lookup_agent, composer_agent],
    tasks=[task_retrieve, task_lookup, task_compose],
    process=Process.sequential,
    verbose=True,
)


def run_crew(
    query: str,
    session_id: str = "default",
    record_id: str | None = None,
) -> CrewResponse:
    """runs the crew pipeline: budget check -> guardrails -> kickoff -> validation -> memory"""
    trace_id = str(uuid.uuid4())

    # 1. Budget check
    budget_check(query)

    # 2. Input guardrails (raises PromptInjectionError on hit; masks PII)
    sanitised_query = apply_input_guardrails(query)

    # 3. Crew kickoff
    inputs = {
        "query": sanitised_query,
        "record_id": record_id or "NONE",
    }
    raw_result = _crew.kickoff(inputs=inputs)

    # 4. Parse raw result into field values
    raw_text = str(raw_result)
    parsed = _parse_crew_output(raw_text, sanitised_query, record_id)

    # 5. Validate with Pydantic
    response = CrewResponse(
        query=sanitised_query,
        answer=parsed["answer"],
        sources=parsed["sources"],
        ticket_status=parsed.get("ticket_status"),
        escalation_score=parsed.get("escalation_score"),
        recommend_escalation=parsed.get("recommend_escalation"),
        grounded=parsed["grounded"],
        session_id=session_id,
        trace_id=trace_id,
    )

    # 6. Store in session memory
    add_turn(session_id, human_text=query, ai_text=response.answer)

    return response


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_crew_output(
    raw_text: str,
    query: str,
    record_id: str | None,
) -> dict:
    """extracts ticket status, scores, and sources from crew output text"""
    from rag.generator import GROUNDEDNESS_THRESHOLD, FALLBACK_ANSWER
    from rag.retriever import retrieve

    # Determine groundedness via retrieval similarity
    chunks = retrieve(query, "ola_sentence", k=1)
    top_score = chunks[0]["score"] if chunks else 0.0
    grounded = top_score >= GROUNDEDNESS_THRESHOLD

    # Extract KB source IDs mentioned in the raw output
    sources = list(set(re.findall(r"KB-\d{3}", raw_text)))

    # Extract ticket fields if a lookup was performed
    ticket_status: str | None = None
    escalation_score: float | None = None
    recommend_escalation: bool | None = None

    if record_id and record_id != "NONE":
        from tools.ticket_tool import get_ticket_dict
        tkt = get_ticket_dict(record_id)
        if "error" not in tkt:
            ticket_status = tkt["status"]
            escalation_score = tkt["escalation_score"]
            recommend_escalation = tkt["recommend_escalation"]

    # Final answer — use the composer's output or fallback
    answer = raw_text.strip() if grounded else FALLBACK_ANSWER

    return {
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
        "ticket_status": ticket_status,
        "escalation_score": escalation_score,
        "recommend_escalation": recommend_escalation,
    }
