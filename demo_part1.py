"""
demo_part1.py - runs tasks 1 to 5 and dumps output trascripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).parent
TRANSCRIPTS = ROOT / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


def capture(fn, *args, **kwargs) -> str:
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def main() -> None:
    # ── Task 1: Dataset validation ────────────────────────────────────────────
    print("=== Task 1: Dataset Validation ===")
    from data.dataset import SUPPORT_TICKETS, validate_tickets
    out1 = capture(validate_tickets, SUPPORT_TICKETS)
    (TRANSCRIPTS / "task1_dataset_validation.txt").write_text(out1, encoding="utf-8")
    print(out1)
    print(f"  → Saved: transcripts/task1_dataset_validation.txt\n")

    # ── Task 3: Build indexes ─────────────────────────────────────────────────
    print("=== Task 3: Building ChromaDB Indexes ===")
    from data.knowledge_base import KB_DOCUMENTS
    from rag.indexer import build_indexes, collection_stats
    out3 = capture(build_indexes, KB_DOCUMENTS)
    print(out3)
    print(f"  Collection stats: {collection_stats()}\n")

    # ── Task 4: RAG demo — ≥5 in-scope + 1 out-of-scope ─────────────────────
    print("=== Task 4: Grounded Generation Demo ===")
    from rag.generator import generate

    demo_queries = [
        ("What is Ola's refund policy for cancelled rides?", "in-scope"),
        ("How are VIP customers handled at Ola?", "in-scope"),
        ("What are the SLA targets for P1 priority issues?", "in-scope"),
        ("When is a ticket automatically escalated?", "in-scope"),
        ("How does Ola communicate during a service outage?", "in-scope"),
        ("What is the best recipe for biryani?", "OUT-OF-SCOPE"),
    ]

    lines = ["Task 4 — Grounded Generation Demo", "=" * 60, ""]
    for query, label in demo_queries:
        result = generate(query)
        lines.append(f"[{label}] {query}")
        lines.append(f"  top_score : {result['top_score']:.4f}")
        lines.append(f"  grounded  : {result['grounded']}")
        lines.append(f"  sources   : {result['sources']}")
        lines.append(f"  answer    : {result['answer'][:200]}")
        lines.append("")
        print(f"  [{label}] top_sim={result['top_score']:.4f}  grounded={result['grounded']}")
        print(f"    Answer: {result['answer'][:120]}")
        print()

    rag_txt = "\n".join(lines)
    (TRANSCRIPTS / "task4_rag_demo.txt").write_text(rag_txt, encoding="utf-8")
    print(f"  → Saved: transcripts/task4_rag_demo.txt\n")

    # ── Task 5: Chunking strategy evaluation ──────────────────────────────────
    print("=== Task 5: Chunking Evaluation ===")
    from evaluation.chunking_eval import print_comparison_report
    out5 = capture(print_comparison_report)
    (TRANSCRIPTS / "task5_chunking_eval.txt").write_text(out5, encoding="utf-8")
    print(out5)
    print(f"  → Saved: transcripts/task5_chunking_eval.txt\n")

    print("✓ Part 1 demos complete.")


if __name__ == "__main__":
    main()
