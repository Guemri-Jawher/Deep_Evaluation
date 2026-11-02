"""Run the hand-written RAG suite: pipeline answers scored by deterministic checks and LLM judges.

Each answerable case costs 1 pipeline call and, with judges, 4 judge calls. Keep --limit small.

Examples:
    uv run python scripts/run_rag_eval.py --no-judges --limit 3
    uv run python scripts/run_rag_eval.py --limit 5 --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # judge text has characters cp1252 lacks

from evaluation.core.report import save_report  # noqa: E402
from evaluation.core.types import SuiteReport  # noqa: E402
from evaluation.hard_coded_lib.suites.rag import run_rag_suite  # noqa: E402

from llm.llm import agent_model  # noqa: E402
from observability.langfuse_client import flush  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.retriever import STRATEGIES, get_retriever  # noqa: E402

JUDGE_KEYS = ("faithfulness", "answer_relevance", "context_relevance", "correctness")


def show(report: SuiteReport) -> None:
    """Print the metric table, then each case with its verdict, failure diagnosis and judge reasons."""
    print(f"\n=== {report.suite}  n={report.n}  config={report.config}")
    for m in report.metrics.values():
        print(f"  {m.name:<20} {m.mean:5.3f} [{m.ci_low:5.3f}, {m.ci_high:5.3f}]  n={m.n}")
    for c in report.cases:
        flag = "PASS" if c.passed else "FAIL"
        judged = " ".join(f"{k}={c.scores[k]:.2f}" for k in JUDGE_KEYS if k in c.scores)
        retr = " ".join(f"{k}={c.scores[k]:.2f}" for k in ("recall@5", "mrr") if k in c.scores)
        cite = f"citation_validity={c.scores.get('citation_validity', float('nan')):.2f}"
        print(f"\n[{flag}] {c.case_id} [{c.category}] {retr} {cite} {judged}".rstrip())
        if c.error:
            print(f"   error: {c.error[:300]}")
        if c.details.get("failure"):
            print(f"   diagnosis: {c.details['failure']}")
        for k in JUDGE_KEYS:
            if f"{k}_reason" in c.details and not c.passed:
                print(f"   {k}: {str(c.details[f'{k}_reason'])[:250]}")
        if "answer" in c.details:
            print(f"   answer: {c.details['answer'][:200]!r}")


def main() -> None:
    """Parse options, run the RAG suite, print the results and save the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=3, help="cases to run (default 3)")
    parser.add_argument("--split", choices=("dev", "heldout"))
    parser.add_argument("--strategy", choices=STRATEGIES, help="retrieval strategy")
    parser.add_argument("--no-judges", action="store_true", help="skip the LLM judges")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-save", action="store_true", help="do not write reports/")
    args = parser.parse_args()

    pipeline = RagPipeline(get_retriever(), agent_model(), strategy=args.strategy)
    report = run_rag_suite(
        pipeline=pipeline,
        use_judges=not args.no_judges,
        split=args.split,
        limit=args.limit,
        max_workers=args.workers,
    )
    show(report)
    if not args.no_save:
        print("saved", save_report(report))


if __name__ == "__main__":
    try:
        main()
    finally:
        flush()  # a short-lived process must push pending traces before exiting
