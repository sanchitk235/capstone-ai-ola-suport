"""
crew/guardrails.py - input and output guardrails.
1. masks indian phone numbers
2. detects prompt injection attempts with regex patetrns
3. groundedness verification using similarity threshold
"""

from __future__ import annotations

import re


class PromptInjectionError(ValueError):
    """thrown when prompt injection attempt detected"""


# matches indian phone numbers (+91 optional, 10 digits starting 6-9)
_PHONE_RE = re.compile(
    r"""
    (?:                         # optional country-code prefix
        \+?91               # +91 or 91
        [\s\-.]?            # optional separator
    )?
    \b                          # word boundary
    [6-9]\d{9}                  # 10-digit Indian mobile (starts 6-9)
    \b
    """,
    re.VERBOSE,
)

PHONE_MASK = "[PHONE REDACTED]"


def mask_pii(text: str) -> str:
    """swaps indian mobile numbers for [PHONE REDACTED]"""
    return _PHONE_RE.sub(PHONE_MASK, text)


# ── 2. Prompt-injection detection ─────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(?:previous\s+)?instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?[\w\s]+(?:AI|bot|model|assistant)", re.I),
    re.compile(r"disregard\s+(?:the\s+|your\s+)?(?:above|previous|system|prior)\b", re.I),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)\b", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?(?:instructions?|training|guidelines?)", re.I),
    re.compile(r"override\s+(?:your\s+)?(?:safety|system|security)\s*(?:guidelines?|rules?|settings?)?", re.I),
]


def detect_injection(text: str) -> bool:
    """checks if query contains prompt injection phrases"""
    return any(pat.search(text) for pat in _INJECTION_PATTERNS)


def check_injection(text: str) -> None:
    """raises PromptInjectionError if query looks suspicious"""
    if detect_injection(text):
        raise PromptInjectionError(
            "Potential prompt-injection detected. Request blocked."
        )


# ── 3. Output-side groundedness check ─────────────────────────────────────────

def groundedness_check(query: str, collection_name: str = "ola_sentence") -> dict:
    """validates whether retrieved docs exceed groundedness threshold"""
    from rag.generator import generate

    return generate(query, collection_name=collection_name)


# ── Combined input guardrail pipeline ─────────────────────────────────────────

def apply_input_guardrails(text: str) -> str:
    """runs injection check then masks phone numbers"""
    check_injection(text)
    return mask_pii(text)
