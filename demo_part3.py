"""
demo_part3.py - runs task 13 15-query evaluation benchmark.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parent
TRANSCRIPTS = ROOT / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


def main() -> None:
    # Build indexes
    from data.knowledge_base import KB_DOCUMENTS
    from rag.indexer import build_indexes
    build_indexes(KB_DOCUMENTS)

    print("=== Task 13: LLM-as-Judge Evaluation (15 queries) ===")
    from evaluation.llm_judge import print_evaluation_report

    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    print_evaluation_report()
    sys.stdout = old
    report = buf.getvalue()

    (TRANSCRIPTS / "task13_evaluation.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Saved: transcripts/task13_evaluation.txt\n")
    print("✓ Part 3 demos complete.")


if __name__ == "__main__":
    main()
