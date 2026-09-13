"""
crew/tasks.py - crewai task definitions.
1. task_retrieve: pulls relevent KB context
2. task_lookup: checks ticket status/escalation if id given
3. task_compose: combines both into final answer
"""

from __future__ import annotations

from crewai import Task

from crew.agents import retrieval_agent, lookup_agent, composer_agent


# Task 1: RAG retrieval

task_retrieve = Task(
    description=(
        "Search Ola's knowledge base for policy information relevant to "
        "the following user query:\n\n"
        "  Query: {query}\n\n"
        "Use the rag_lookup tool with the exact query above. "
        "Return the answer text and the source KB document IDs."
    ),
    expected_output=(
        "A passage of Ola policy text that directly answers the query, "
        "together with a list of source KB document IDs (e.g. KB-001, KB-004)."
    ),
    agent=retrieval_agent,
)

# ── Task 2: Look up support ticket ────────────────────────────────────────────

task_lookup = Task(
    description=(
        "Look up the Ola support ticket with the following record ID:\n\n"
        "  Record ID: {record_id}\n\n"
        "Use the check_support_ticket_status tool. "
        "If the record_id is 'NONE' or blank, respond with: "
        "'No ticket lookup required for this query.' and stop."
    ),
    expected_output=(
        "The ticket's current status, resolution_time_hours, escalation_score "
        "(0–1), and a YES/NO escalation recommendation. "
        "If record_id is NONE, output 'No ticket lookup required.'"
    ),
    agent=lookup_agent,
)

# ── Task 3: Compose the final user-facing response ────────────────────────────

task_compose = Task(
    description=(
        "Compose a clear, grounded, and concise final response for the user.\n\n"
        "  Original query : {query}\n"
        "  Record ID      : {record_id}\n\n"
        "Use ONLY the context provided by the Retrieval and Lookup tasks above. "
        "Do NOT invent information that was not supplied by those tasks. "
        "If no KB context was found, state that clearly. "
        "If a ticket was looked up, include its status and escalation recommendation."
    ),
    expected_output=(
        "A single, well-structured response (3–6 sentences) that directly "
        "answers the user's query using only grounded information from the "
        "retrieval and lookup tasks."
    ),
    agent=composer_agent,
    context=[task_retrieve, task_lookup],   # receives outputs of both prior tasks
)
