"""
data/dataset.py - generates synthetic ola support tickets.
SEED=42, 50 rows total.
res time between 1 and 72 hrs (matches real life ola dispute SLAs).
escalated prob 0.20 to stay in the 10-30% range requried by spec.
"""

import json
import random
from collections import Counter
from pathlib import Path

# basic setup constants

SEED = 42
NUM_RECORDS = 50

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]
CAT_WEIGHTS = [25, 20, 20, 15, 20]

STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
STA_WEIGHTS = [30, 25, 15, 20, 10]

ESCALATED_PROB = 0.20
RES_TIME_MIN = 1.0
RES_TIME_MAX = 72.0
MAX_DAYS = 30


# quick generator func
def generate_tickets(seed: int = SEED, n: int = NUM_RECORDS) -> list[dict]:
    """generate list of fake tickets with deterministic seed"""
    rng = random.Random(seed)
    records: list[dict] = []
    for i in range(1, n + 1):
        category = rng.choices(CATEGORIES, weights=CAT_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STA_WEIGHTS, k=1)[0]
        resolution_time_hours = round(rng.uniform(RES_TIME_MIN, RES_TIME_MAX), 1)
        days_since_created = rng.randint(0, MAX_DAYS)
        escalated = rng.random() < ESCALATED_PROB
        records.append(
            {
                "record_id": f"TKT-{i:03d}",
                "category": category,
                "status": status,
                "resolution_time_hours": resolution_time_hours,
                "days_since_created": days_since_created,
                "escalated": escalated,
            }
        )
    return records


# dataset exported for other modules
SUPPORT_TICKETS: list[dict] = generate_tickets()

# fast id lookup
TICKET_LOOKUP: dict[str, dict] = {t["record_id"]: t for t in SUPPORT_TICKETS}


def validate_tickets(records: list[dict]) -> None:
    """checks that ticket distribution matches all constraints"""
    categories = Counter(r["category"] for r in records)
    statuses = Counter(r["status"] for r in records)
    escalated_count = sum(1 for r in records if r["escalated"])
    escalated_pct = escalated_count / len(records)

    separator = "=" * 50
    print(f"\n{separator}")
    print("  Dataset Validation Report")
    print(separator)
    print(f"  Total records : {len(records)}")

    print("\n  Category counts (need ≥3 each):")
    for cat in CATEGORIES:
        count = categories.get(cat, 0)
        ok = "✓" if count >= 3 else "✗ FAIL"
        print(f"    {cat:<20} {count:>3}  {ok}")

    print("\n  Status counts (need ≥1 each):")
    for sta in STATUSES:
        count = statuses.get(sta, 0)
        ok = "✓" if count >= 1 else "✗ FAIL"
        print(f"    {sta:<20} {count:>3}  {ok}")

    band_ok = 0.10 <= escalated_pct <= 0.30
    band_flag = "✓" if band_ok else "✗ FAIL — outside [10 %, 30 %]"
    print(
        f"\n  Escalated: {escalated_count}/{len(records)} = "
        f"{escalated_pct:.1%}  {band_flag}"
    )

    # sanity checks - throw error if seed gave bad distribution
    for cat in CATEGORIES:
        assert categories.get(cat, 0) >= 3, (
            f"Category '{cat}' has {categories.get(cat, 0)} records (need ≥3)"
        )
    for sta in STATUSES:
        assert statuses.get(sta, 0) >= 1, (
            f"Status '{sta}' has 0 records (need ≥1)"
        )
    assert band_ok, (
        f"Escalated % {escalated_pct:.1%} is outside required [10 %, 30 %] band. "
        "Change SEED or ESCALATED_PROB and regenerate."
    )

    print("\n  ✓ All validations passed.")
    print(separator + "\n")


if __name__ == "__main__":
    validate_tickets(SUPPORT_TICKETS)

    out_path = Path(__file__).parent / "tickets.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(SUPPORT_TICKETS, fh, indent=2)
    print(f"Saved {len(SUPPORT_TICKETS)} records → {out_path}")
