"""
crew/agents.py - crewai agent defintions.
3 agents:
1. retrieval_agent (rag_lookup only)
2. lookup_agent (check_support_ticket_status only - least autonomy)
3. composer_agent (no tools)
"""

from __future__ import annotations

import os

# kill telemetry before importing crewai
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent

from mock_llm.mock_llm import MockLLM
from tools.rag_tool import rag_lookup
from tools.ticket_tool import check_support_ticket_status

_llm = MockLLM()

# Agent 1: policy searcher
retrieval_agent = Agent(
    role="Policy Retrieval Specialist",
    goal=(
        "Search Ola's support knowledge base and return the most relevant "
        "policy information for the user's query."
    ),
    backstory=(
        "You are a specialist in Ola's internal policy documents. "
        "Your sole responsibility is to find and surface the most relevant "
        "KB excerpts for any incoming support question."
    ),
    tools=[rag_lookup],
    llm=_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 2: ticket lookup (only one holding ticket tool)
lookup_agent = Agent(
    role="Ticket Lookup Specialist",
    goal=(
        "Retrieve the current status and escalation score for a support "
        "ticket when provided with a record ID."
    ),
    backstory=(
        "You have exclusive access to Ola's support-ticket database. "
        "You look up tickets by record ID and report their status, "
        "resolution time, and computed escalation score."
    ),
    tools=[check_support_ticket_status],  # lookup only
    llm=_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 3: synthesises final text
composer_agent = Agent(
    role="Response Composer",
    goal=(
        "Combine the retrieved policy context and ticket information into "
        "a single clear, grounded, and accurate response for the user."
    ),
    backstory=(
        "You are the final-stage editor of Ola's support crew. "
        "You receive structured outputs from the Retrieval and Lookup agents "
        "and synthesise them into a polished, user-facing answer. "
        "You never invent information not provided by the other agents."
    ),
    tools=[],   # no tools for composer
    llm=_llm,
    verbose=True,
    allow_delegation=False,
)
