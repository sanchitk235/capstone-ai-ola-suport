"""
evaluation/llm_judge.py - 15 query benchmark testing 4 dimensions.
accuarcy, grounding, completeness, and safety metrics.
covers all 12 KB topics + edge cases and out of scope queries.
"""

from __future__ import annotations

from rag.generator import generate, GROUNDEDNESS_THRESHOLD
from crew.guardrails import detect_injection

# 15 query eval suite (in scope, out of scope, edge case)

TEST_QUERIES: list[dict] = [
    # ── 12 in-scope (one per KB topic) ───────────────────────────────────────
    {
        "id": "Q01",
        "query": "How does Ola classify the priority of a support ticket?",
        "scope": "in",
        "kb_topic": "ticket_priority_classification",
    },
    {
        "id": "Q02",
        "query": "What are the SLA targets for P1 priority issues?",
        "scope": "in",
        "kb_topic": "sla_by_severity",
    },
    {
        "id": "Q03",
        "query": "When is a support ticket escalated to a higher support tier?",
        "scope": "in",
        "kb_topic": "escalation_matrix",
    },
    {
        "id": "Q04",
        "query": "What is Ola's refund policy for cancelled or overcharged rides?",
        "scope": "in",
        "kb_topic": "refund_compensation_policy",
    },
    {
        "id": "Q05",
        "query": "How can customers contact Ola support?",
        "scope": "in",
        "kb_topic": "customer_communication_channel_policy",
    },
    {
        "id": "Q06",
        "query": "Does Ola provide support on public holidays?",
        "scope": "in",
        "kb_topic": "business_hours_holiday_support",
    },
    {
        "id": "Q07",
        "query": "What happens when the same customer complains about the same issue repeatedly?",
        "scope": "in",
        "kb_topic": "repeat_complaint_handling",
    },
    {
        "id": "Q08",
        "query": "How do Ola service credits work and when are they issued?",
        "scope": "in",
        "kb_topic": "service_credit_policy",
    },
    {
        "id": "Q09",
        "query": "How does Ola collect customer feedback after a ride?",
        "scope": "in",
        "kb_topic": "feedback_collection_process",
    },
    {
        "id": "Q10",
        "query": "Are VIP customers given priority support at Ola?",
        "scope": "in",
        "kb_topic": "vip_customer_handling",
    },
    {
        "id": "Q11",
        "query": "How does Ola notify users during a service outage?",
        "scope": "in",
        "kb_topic": "outage_communication_protocol",
    },
    {
        "id": "Q12",
        "query": "How long does Ola keep closed support ticket records?",
        "scope": "in",
        "kb_topic": "data_retention_policy",
    },
    # ── 2 out-of-scope queries ────────────────────────────────────────────────
    {
        "id": "Q13",
        "query": "What is Ola's policy on driver background checks and vehicle inspections?",
        "scope": "out",
        "kb_topic": "N/A — not in knowledge base",
    },
    {
        "id": "Q14",
        "query": "Who won the ICC Cricket World Cup in 2023?",
        "scope": "out",
        "kb_topic": "N/A — completely unrelated",
    },
    # ── 1 edge-case query (multi-tool / borderline) ───────────────────────────
    {
        "id": "Q15",
        "query": (
            "My ticket TKT-001 has been open for a long time and I have not "
            "received any response. Should it be escalated?"
        ),
        "scope": "edge",
        "kb_topic": "escalation_matrix + ticket lookup",
    },
]


# ── Scoring functions ─────────────────────────────────────────────────────────

def _score_accuracy(result: dict, scope: str) -> float:
    """
    1.0 if the system behaved correctly:
      in-scope  → grounded answer (not fallback)
      out-of-scope → fallback returned
      edge      → any answer (partial credit)
    """
    is_fallback = not result["grounded"]
    if scope == "in":
        return 1.0 if not is_fallback else 0.0
    if scope == "out":
        return 1.0 if is_fallback else 0.0
    # edge — give full credit if answer exists
    return 1.0 if result["answer"] else 0.0


def _score_grounding(result: dict, scope: str) -> float:
    """
    1.0 if top retrieval similarity ≥ threshold OR it's correctly out-of-scope.
    """
    if scope == "out" and not result["grounded"]:
        return 1.0  # correctly identified as out-of-scope
    return 1.0 if result["grounded"] else 0.0


def _score_completeness(result: dict, scope: str) -> float:
    """
    1.0 if answer ≥ 80 chars, 0.5 if ≥ 40 chars, else 0.0.
    Out-of-scope fallback is always complete (1.0).
    """
    if scope == "out":
        return 1.0  # fallback message is a complete response
    length = len(result["answer"])
    if length >= 80:
        return 1.0
    if length >= 40:
        return 0.5
    return 0.0


def _score_safety(result: dict) -> float:
    """1.0 if no injection pattern detected in the answer."""
    return 0.0 if detect_injection(result["answer"]) else 1.0


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(
    collection_name: str = "ola_sentence",
) -> list[dict]:
    """Run the full 15-query evaluation and return scored rows."""
    rows: list[dict] = []
    for item in TEST_QUERIES:
        result = generate(item["query"], collection_name=collection_name)
        acc = _score_accuracy(result, item["scope"])
        grd = _score_grounding(result, item["scope"])
        cmp = _score_completeness(result, item["scope"])
        saf = _score_safety(result)
        rows.append(
            {
                "id": item["id"],
                "query": item["query"][:60] + ("…" if len(item["query"]) > 60 else ""),
                "scope": item["scope"],
                "kb_topic": item["kb_topic"],
                "top_score": result["top_score"],
                "grounded": result["grounded"],
                "accuracy": acc,
                "grounding": grd,
                "completeness": cmp,
                "safety": saf,
            }
        )
    return rows


def print_evaluation_report(rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = run_evaluation()

    print("\n" + "=" * 80)
    print("  LLM-as-Judge Evaluation Report — 15 Queries (Task 13)")
    print("=" * 80)

    header = (
        f"\n{'ID':<4} {'Scope':<5} {'TopSim':>7} {'Acc':>5} "
        f"{'Grd':>5} {'Cmp':>5} {'Saf':>5}  Query"
    )
    print(header)
    print("-" * 80)

    for r in rows:
        print(
            f"{r['id']:<4} {r['scope']:<5} {r['top_score']:>7.3f} "
            f"{r['accuracy']:>5.1f} {r['grounding']:>5.1f} "
            f"{r['completeness']:>5.1f} {r['safety']:>5.1f}  "
            f"{r['query']}"
        )

    # Averages
    def avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 3)

    print("-" * 80)
    print(
        f"{'MEAN':<4} {'':<5} {'':>7} "
        f"{avg('accuracy'):>5.3f} {avg('grounding'):>5.3f} "
        f"{avg('completeness'):>5.3f} {avg('safety'):>5.3f}"
    )
    print("\n")
    print(f"  Average Accuracy    : {avg('accuracy')}")
    print(f"  Average Grounding   : {avg('grounding')}")
    print(f"  Average Completeness: {avg('completeness')}")
    print(f"  Average Safety      : {avg('safety')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_evaluation_report()
