"""Run the DeepEval suites (LLM-judged) on a few golden questions and print scores and judge reasons.

Needs an LLM key for the provider chosen by LLM_PROVIDER. Each case costs several judge calls per metric, so keep --limit small.

Examples:
    uv run python scripts/try_deepeval.py
    uv run python scripts/try_deepeval.py --suite rag --limit 5 --split dev
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
from evaluation.deep_eval.rag_eval.suite import (  # noqa: E402
    run_deepeval_rag_suite,
    run_deepeval_retrieval_suite,
)

from llm.llm import agent_model, judge_model  # noqa: E402
from observability.langfuse_client import flush  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.retriever import STRATEGIES, get_retriever  # noqa: E402


def show(report: SuiteReport) -> None:
    """Print the metric table, then every case with its scores and the reasons of failed metrics."""
    print(f"\n=== {report.suite}  n={report.n}  config={report.config}")
    for m in report.metrics.values():
        print(f"  {m.name:<28} {m.mean:5.3f} [{m.ci_low:5.3f}, {m.ci_high:5.3f}]  n={m.n}")
    for c in report.cases:
        flag = "ok  " if c.passed else "FAIL"
        scores = " ".join(f"{k.removeprefix('de_')}={v:.2f}" for k, v in c.scores.items())
        print(f"\n[{flag}] {c.case_id} [{c.category}] {scores}")
        if c.error:
            print(f"   error: {c.error[:300]}")
        for name in c.details.get("failed_metrics", []):
            print(f"   {name}: {str(c.details['reasons'].get(name, ''))[:300]}")
        if "answer" in c.details:
            print(f"   answer: {c.details['answer'][:200]!r}")


def main() -> None:
    """Parse options, run the chosen DeepEval suite(s), print the results and save the reports."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--suite", choices=("retrieval", "rag", "both"), default="both")
    parser.add_argument("--limit", type=int, default=3, help="cases per suite (default 3)")
    parser.add_argument("--split", choices=("dev", "heldout"))
    parser.add_argument(
        "--strategy", choices=STRATEGIES, help="retrieval strategy (default: configured)"
    )
    parser.add_argument("-k", type=int, default=5, help="top-k chunks for the retrieval suite")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-save", action="store_true", help="do not write reports/")
    args = parser.parse_args()

    judge = judge_model()
    reports: list[SuiteReport] = []
    if args.suite in ("retrieval", "both"):
        reports.append(
            run_deepeval_retrieval_suite(
                judge,
                strategy=args.strategy,
                k=args.k,
                split=args.split,
                limit=args.limit,
                max_workers=args.workers,
            )
        )
    if args.suite in ("rag", "both"):
        pipeline = RagPipeline(get_retriever(), agent_model(), strategy=args.strategy)
        reports.append(
            run_deepeval_rag_suite(
                judge,
                pipeline=pipeline,
                split=args.split,
                limit=args.limit,
                max_workers=args.workers,
            )
        )

    for rep in reports:
        show(rep)
        if not args.no_save:
            print("saved", save_report(rep))


if __name__ == "__main__":
    try:
        main()
    finally:
        flush()  # a short-lived process must push pending traces before exiting
