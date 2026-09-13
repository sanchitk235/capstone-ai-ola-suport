"""
mock_llm/mock_llm.py - zero network mock llm for crewai react loop.
handles 2 edge cases:
1. skips system prompt when looking for 'Observation:' (react template has that word)
2. dispatches tools by schema arg names (record_id or query) rather than string matching names
"""

from __future__ import annotations

import re
from typing import Any, Optional

from crewai.llms.base_llm import BaseLLM


# simple helper for rag generator
def get_mock_response(query: str, context: str = "") -> str:
    """quick mock answer echoing context"""
    if context:
        clean_ctx = context[:400].replace("\n", " ")
        return (
            f"Based on Ola's support policies, here is the relevant information: "
            f"{clean_ctx} "
            f"This addresses your question: {query[:120]}."
        )
    return f"I don't have sufficient context to answer: {query[:120]}"


class MockLLM(BaseLLM):
    """deterministic mock llm subclassing crewai BaseLLM with ReAct format"""

    model: str = "mock/mock-v1"
    temperature: float = 0.0
    max_tokens: int = 512

    def __init__(self, model: str = "mock/mock-v1", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    # ── Core call ─────────────────────────────────────────────────────────────

    def call(
        self,
        messages: list[dict[str, Any]],
        callbacks: Optional[list] = None,
        available_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        """produces mock completion for crewai react loop"""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        # messages after task are observations
        post_task_user = user_msgs[1:]

        has_observation = any(
            "Observation:" in m.get("content", "")
            for m in post_task_user
        )

        if has_observation:
            return self._final_answer_from_observation(post_task_user)

        # check if agent has tools to execute
        task_text = user_msgs[0]["content"] if user_msgs else ""
        tools = available_functions or []

        if tools:
            return self._dispatch_tool(tools, task_text)

        # composer path (no tools)
        return (
            "Thought: I have all context needed to compose the final response.\n"
            f"Final Answer: Based on Ola's support policies and the retrieved "
            f"ticket information, here is a comprehensive response: {task_text[:280]}"
        )

    def _final_answer_from_observation(
        self, post_task_user: list[dict]
    ) -> str:
        """extracts last tool result and builds final answer"""
        obs_text = ""
        for m in reversed(post_task_user):
            if "Observation:" in m.get("content", ""):
                obs_text = m["content"].split("Observation:", 1)[-1].strip()
                break
        return (
            "Thought: I now have all the information needed to answer "
            "the user's question.\n"
            f"Final Answer: {obs_text[:480]}"
        )

    def _dispatch_tool(self, tools: list, task_text: str) -> str:
        """dispatches tool by checking schema arg names rather than func name"""
        for tool in tools:
            schema_keys = self._schema_keys(tool)

            if "record_id" in schema_keys:
                # ticket lookup
                record_id = self._extract_tkt_id(task_text) or "TKT-001"
                return (
                    "Thought: I need to look up the support ticket in the "
                    "database to retrieve its current status and escalation score.\n"
                    "Action: check_support_ticket_status\n"
                    f'Action Input: {{"record_id": "{record_id}"}}'
                )

            if "query" in schema_keys:
                # rag kb search
                clean = task_text[:140].replace("\n", " ").replace('"', "'")
                return (
                    "Thought: I need to search the Ola knowledge base for "
                    "relevant policy information.\n"
                    "Action: rag_lookup\n"
                    f'Action Input: {{"query": "{clean}"}}'
                )

        # fallback if schema not recognised
        return (
            "Thought: Could not identify the right tool; returning a direct answer.\n"
            f"Final Answer: {task_text[:200]}"
        )

    @staticmethod
    def _schema_keys(tool) -> set[str]:
        """grabs param names from tool args_schema or json schema dict"""
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                return set(tool.args_schema.model_fields.keys())
            except AttributeError:
                pass

        if isinstance(tool, dict):
            props = (
                tool.get("function", {})
                .get("parameters", {})
                .get("properties", {})
            )
            return set(props.keys())

        return set()

    @staticmethod
    def _extract_tkt_id(text: str) -> Optional[str]:
        """extracts TKT-NNN id from query text"""
        match = re.search(r"TKT-\d{3}", text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def supports_function_calling(self) -> bool:
        """use react text output instead of json function calling"""
        return False

    def get_context_window_size(self) -> int:
        return 8192
