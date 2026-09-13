"""
demo_part4.py - runs tasks 14 to 16 for autogen review, governance, and cache.
"""

from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

ROOT = Path(__file__).parent
TRANSCRIPTS = ROOT / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


def capture(fn, *args, **kwargs) -> str:
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


def main() -> None:
    # Build indexes
    from data.knowledge_base import KB_DOCUMENTS
    from rag.indexer import build_indexes
    build_indexes(KB_DOCUMENTS)

    # ── Task 14: AutoGen review ───────────────────────────────────────────────
    print("=== Task 14: AutoGen Review Stage Demo ===")
    from review.autogen_review import run_review

    lines = ["Task 14 — AutoGen RoundRobin Review Demo", "=" * 60, ""]

    CONTEXT = (
        "[KB-004] Ola processes refunds for verified overcharges and ride cancellations "
        "caused by driver no-show or confirmed technical failures. Refunds are returned "
        "within 5–7 business days."
    )

    # Case 1: Approve (grounded draft)
    good_draft = (
        "Ola processes refunds for verified overcharges and ride cancellations "
        "caused by driver no-show or confirmed technical failures on the platform. "
        "Refunds are credited to the original payment method within 5–7 business days."
    )
    print("  Case 1: Approve (grounded draft) …")
    verdict1 = run_review(draft=good_draft, context=CONTEXT, inject_flaw=False)
    lines += [
        "=== Case 1: Grounded Draft (Approve) ===",
        f"Draft   : {good_draft[:200]}",
        f"approved: {verdict1.approved}",
        f"reason  : {verdict1.reason}",
        f"final   : {verdict1.final_answer[:200]}",
        "",
    ]
    print(f"    approved={verdict1.approved}")

    # Case 2: Revise (ungrounded claim injected)
    bad_draft = (
        "Ola offers a 100% refund always, no questions asked, for all ride issues. "
        "Refunds are processed within 5–7 business days."
    )
    print("  Case 2: Revise (ungrounded claim injected) …")
    verdict2 = run_review(draft=bad_draft, context=CONTEXT, inject_flaw=True)
    lines += [
        "=== Case 2: Ungrounded Draft (Revise) ===",
        f"Draft   : {bad_draft[:200]}",
        f"approved: {verdict2.approved}",
        f"reason  : {verdict2.reason}",
        f"final   : {verdict2.final_answer[:200]}",
        "",
    ]
    print(f"    approved={verdict2.approved}")

    t14 = "\n".join(lines)
    (TRANSCRIPTS / "task14_autogen_demo.txt").write_text(t14, encoding="utf-8")
    print(f"  → Saved: transcripts/task14_autogen_demo.txt\n")

    # ── Task 15a: Least-autonomy ──────────────────────────────────────────────
    print("=== Task 15: Governance Demo ===")
    from governance.least_autonomy import demonstrate_least_autonomy
    out15a = capture(demonstrate_least_autonomy)

    # Task 15b: Budget cap
    from governance.risk_classification import budget_check, BudgetExceededError, RISK_LEVEL, RISK_JUSTIFICATION
    budget_lines = [
        "",
        "=== Runtime Budget Cap Demo ===",
        f"MAX_TOKENS_PER_REQUEST = 500",
        "",
    ]
    oversized = "What is Ola's policy? " * 50  # ~400 words → >500 tokens estimated
    try:
        budget_check(oversized)
        budget_lines.append("No error raised (unexpected).")
    except BudgetExceededError as exc:
        budget_lines.append(f"BudgetExceededError raised: {exc}")
        budget_lines.append("  → Server would return HTTP 429.")

    budget_lines += [
        "",
        "=== Risk Classification ===",
        f"Risk Level: {RISK_LEVEL}",
        RISK_JUSTIFICATION,
    ]

    t15 = out15a + "\n".join(budget_lines)
    (TRANSCRIPTS / "task15_governance_demo.txt").write_text(t15, encoding="utf-8")
    print(out15a[:300])
    print("\n".join(budget_lines))
    print(f"  → Saved: transcripts/task15_governance_demo.txt\n")

    # ── Task 16: Cache demo ───────────────────────────────────────────────────
    print("=== Task 16: Response Cache Demo ===")
    from governance.cache import cached_generate, cache_stats, clear_cache

    clear_cache()
    CACHE_QUERY = "What is Ola's refund policy for cancelled rides?"

    r1 = cached_generate(CACHE_QUERY)
    r2 = cached_generate(CACHE_QUERY)    # identical query — should hit cache
    r3 = cached_generate(CACHE_QUERY.upper())  # normalised to same key

    lines = [
        "Task 16 — Response Cache Demo",
        "=" * 60,
        "",
        f"Query: '{CACHE_QUERY}'",
        "",
        f"Call 1 — cache_hit={r1['cache_hit']}  llm_call_count={r1['llm_call_count']}",
        f"  answer[:80]: {r1['answer'][:80]}",
        "",
        f"Call 2 (identical query) — cache_hit={r2['cache_hit']}  llm_call_count={r2['llm_call_count']}",
        "  (llm_call_count unchanged → no redundant LLM call)",
        "",
        f"Call 3 (upper-cased query, same key) — cache_hit={r3['cache_hit']}  llm_call_count={r3['llm_call_count']}",
        "",
        f"Final cache stats: {cache_stats()}",
    ]

    t16 = "\n".join(lines)
    (TRANSCRIPTS / "task16_cache_demo.txt").write_text(t16, encoding="utf-8")
    print(t16)
    print(f"\n  → Saved: transcripts/task16_cache_demo.txt\n")

    print("✓ Part 4 demos complete.")


if __name__ == "__main__":
    main()
