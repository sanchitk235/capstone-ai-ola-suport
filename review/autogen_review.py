"""
review/autogen_review.py - autogen 0.4.x round robin review team.
Team: PolicyComplianceReviewer + FinalEditor
Uses MaxMessageTermination(3) so task msg + both agents get processed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from pydantic import BaseModel


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class _MockAutogenClient:
    """mock completion client for autogen 0.4.x protocol"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self._total_prompt = 0
        self._total_completion = 0

    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        json_output: bool | None = None,
        extra_create_args: dict[str, Any] | None = None,
        cancellation_token: Any = None,
    ):
        from autogen_core.models import CreateResult, RequestUsage

        content = self._responses[self._index % len(self._responses)]
        self._index += 1
        tokens = len(content.split())
        usage = RequestUsage(prompt_tokens=15, completion_tokens=tokens)
        self._total_prompt += 15
        self._total_completion += tokens
        return CreateResult(
            content=content,
            usage=usage,
            finish_reason="stop",
            cached=False,
            logprobs=None,
        )

    async def create_stream(self, messages, **kwargs):
        raise NotImplementedError("Streaming not supported by MockAutogenClient")

    def actual_usage(self):
        from autogen_core.models import RequestUsage
        return RequestUsage(prompt_tokens=15, completion_tokens=20)

    def total_usage(self):
        from autogen_core.models import RequestUsage
        return RequestUsage(
            prompt_tokens=self._total_prompt,
            completion_tokens=self._total_completion,
        )

    def count_tokens(self, messages, *, tools=()) -> int:
        return sum(len(str(m)) for m in messages) // 4

    @property
    def model_info(self) -> dict:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }


# ── Deterministic response factory ────────────────────────────────────────────

def _make_responses(draft: str, inject_flaw: bool) -> tuple[list[str], list[str]]:
    """
    Return (reviewer_responses, editor_responses) for the given scenario.
    inject_flaw=True simulates a draft with an ungrounded claim that the
    reviewer flags and the editor corrects.
    """
    if inject_flaw:
        # ── Revision case: draft contains "100% refund always" (ungrounded) ─
        reviewer_response = (
            "I have reviewed the draft answer. "
            "The claim 'Ola offers a 100% refund always' is ungrounded — "
            "KB-004 states refunds are case-specific and do not guarantee 100%. "
            "The FinalEditor should revise this claim before approving."
        )
        editor_verdict = json.dumps({
            "approved": False,
            "final_answer": (
                "Ola processes refunds for verified overcharges and ride cancellations "
                "caused by driver no-show or confirmed technical failures. "
                "Refunds are credited within 5–7 business days and are assessed "
                "case-by-case; a blanket 100% refund guarantee does not exist in "
                "Ola's policy. (Source: KB-004)"
            ),
            "reason": (
                "Removed the ungrounded claim 'Ola offers a 100% refund always'. "
                "Replaced with accurate policy language from KB-004."
            ),
        })
    else:
        # ── Approval case: draft is grounded and accurate ──────────────────
        reviewer_response = (
            "I have reviewed the draft answer. "
            "The content is consistent with KB-004 (refund policy) and "
            "accurately reflects Ola's refund timelines and conditions. "
            "No policy violations or ungrounded claims detected. "
            "Recommending approval unchanged."
        )
        editor_verdict = json.dumps({
            "approved": True,
            "final_answer": draft,
            "reason": (
                "Draft is fully grounded in KB-004. "
                "Refund timelines and conditions are accurately stated. "
                "No revisions required."
            ),
        })

    return [reviewer_response], [editor_verdict]


# ── Public API ────────────────────────────────────────────────────────────────

async def _run_review_async(
    draft: str,
    context: str,
    inject_flaw: bool = False,
) -> ReviewVerdict:
    """executes autogen review team asynchrnously"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.teams import RoundRobinGroupChat

    reviewer_responses, editor_responses = _make_responses(draft, inject_flaw)

    reviewer_client = _MockAutogenClient(reviewer_responses)
    editor_client = _MockAutogenClient(editor_responses)

    policy_reviewer = AssistantAgent(
        name="PolicyComplianceReviewer",
        system_message=(
            "You are an Ola policy-compliance reviewer. "
            "Check the draft answer for factual accuracy against the provided context. "
            "Flag any claims not supported by the context."
        ),
        model_client=reviewer_client,
    )

    final_editor = AssistantAgent(
        name="FinalEditor",
        system_message=(
            "You are the final editor. Based on the reviewer's assessment, "
            "either approve the draft unchanged or revise it. "
            "Return a JSON object with fields: approved, final_answer, reason."
        ),
        model_client=editor_client,
        output_content_type=StructuredMessage[ReviewVerdict],
    )

    team = RoundRobinGroupChat(
        participants=[policy_reviewer, final_editor],
        termination_condition=MaxMessageTermination(3),
        # MaxMessageTermination(3): counts task message (1) + reviewer (2) + editor (3)
        # → both agents speak exactly once.
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )

    task_message = (
        f"Please review the following draft answer.\n\n"
        f"CONTEXT (from knowledge base):\n{context}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        "Reviewer: check for ungrounded claims. "
        "Editor: produce a ReviewVerdict JSON."
    )

    result = await team.run(task=task_message)

    # Extract the last message from the FinalEditor as the verdict
    verdict_json: str | None = None
    for msg in reversed(result.messages):
        content = getattr(msg, "content", None)
        if isinstance(content, ReviewVerdict):
            return content
        if isinstance(content, str):
            try:
                data = json.loads(content)
                return ReviewVerdict(**data)
            except Exception:
                continue

    # Fallback if parsing failed
    return ReviewVerdict(
        approved=True,
        final_answer=draft,
        reason="Review stage completed; verdict parsing used fallback.",
    )


def run_review(
    draft: str,
    context: str,
    inject_flaw: bool = False,
) -> ReviewVerdict:
    """synchronous wrapper to execute review team"""
    return asyncio.run(_run_review_async(draft, context, inject_flaw))
