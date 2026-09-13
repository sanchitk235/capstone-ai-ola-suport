"""
tools/ticket_tool.py - crewai tool to check ticket status.
formula: escalation_score = 0.6 * escalated + 0.4 * (days / 30)
cutoff is 0.33 based on 80th percentile days for unescalated tickets (~25 days).
"""

from __future__ import annotations

from crewai.tools import tool

from data.dataset import TICKET_LOOKUP

MAX_DAYS: float = 30.0
ESCALATION_THRESHOLD: float = 0.33


def _compute_escalation_score(ticket: dict) -> float:
    """calulate escalation score clamped between 0 and 1"""
    recency = ticket["days_since_created"] / MAX_DAYS
    score = 0.6 * float(ticket["escalated"]) + 0.4 * recency
    return round(min(1.0, max(0.0, score)), 4)


@tool("check_support_ticket_status")
def check_support_ticket_status(record_id: str) -> str:
    """fetches ticket info and recommends if escalation is needed"""
    ticket = TICKET_LOOKUP.get(record_id.upper().strip())
    if ticket is None:
        return (
            f"Ticket {record_id!r} was not found in the support database. "
            "Please verify the record ID and try again."
        )

    score = _compute_escalation_score(ticket)
    recommend = score >= ESCALATION_THRESHOLD

    return (
        f"Ticket ID       : {ticket['record_id']}\n"
        f"Category        : {ticket['category']}\n"
        f"Status          : {ticket['status']}\n"
        f"Resolution time : {ticket['resolution_time_hours']} hours\n"
        f"Days since created : {ticket['days_since_created']}\n"
        f"Escalated flag  : {ticket['escalated']}\n"
        f"Escalation score: {score} (threshold = {ESCALATION_THRESHOLD})\n"
        f"Recommend escalation: {'YES' if recommend else 'NO'}"
    )


def get_ticket_dict(record_id: str) -> dict:
    """helper returning ticket dict with escalation score for api/eval"""
    ticket = TICKET_LOOKUP.get(record_id.upper().strip())
    if ticket is None:
        return {"error": f"Ticket {record_id!r} not found."}
    score = _compute_escalation_score(ticket)
    return {
        **ticket,
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
    }
