"""
governance/risk_classification.py - risk clasification and token budget cap.
Medium risk: handles support tickets and refund recommendations with business impact.
Token budget cap: 500 tokens per request (throws BudgetExceededError on violation).
"""

from __future__ import annotations

MAX_TOKENS_PER_REQUEST: int = 500
_WORDS_TO_TOKENS_FACTOR: float = 1.3


class BudgetExceededError(RuntimeError):
    """thrown when input query exceeds token budget cap"""

    def __init__(self, estimated: float, cap: int) -> None:
        self.estimated = estimated
        self.cap = cap
        super().__init__(
            f"Request rejected: estimated {estimated:.0f} tokens exceeds "
            f"budget cap of {cap} tokens."
        )


def budget_check(query: str) -> None:
    """estimates tokens and blocks requests over 500 token limit"""
    word_count = len(query.split())
    estimated_tokens = word_count * _WORDS_TO_TOKENS_FACTOR
    if estimated_tokens > MAX_TOKENS_PER_REQUEST:
        raise BudgetExceededError(estimated=estimated_tokens, cap=MAX_TOKENS_PER_REQUEST)


# ── Risk classification summary (also in README) ──────────────────────────────

RISK_LEVEL = "Medium"

RISK_JUSTIFICATION = (
    "The Ola Domain Support Agent is classified as MEDIUM risk under the "
    "Low / Medium / High risk taxonomy. It processes customer support tickets "
    "containing personal data, makes policy-driven recommendations that affect "
    "refund eligibility and escalation decisions, and provides outputs that "
    "directly influence both customer-facing and agent-facing workflows. "
    "It does not handle medical data, hiring decisions, or financial instruments, "
    "so High risk does not apply. Pure Low-risk summarisation systems make no "
    "consequential recommendations, whereas this agent's refund and escalation "
    "guidance has tangible business and customer impact. Mitigations include: "
    "output groundedness guardrails, Pydantic schema validation, an independent "
    "AutoGen review stage, least-autonomy tool isolation, and this runtime "
    "budget cap."
)
