"""Run the retrieval evaluation (no LLM, no API key) and print a per-strategy summary.

Evaluates how well `vector`, `bm25` and `hybrid` retrieval find the gold evidence for each
answerable question in `datasets/data/v1/rag_golden.jsonl` (unanswerable rows are skipped).
What it does:
    1. Builds or loads the index (Chroma + BM25) through `rag.retriever`.
    2. For each case, retrieves the top `k` chunks and scores them with `recall@k`, `hit@k`,
       `precision@5`, `mrr` and `ndcg@5` (`evaluation.metrics.retrieval`). A case passes when
       `recall@5` is 1.0, i.e. all its evidence units are in the top 5.
    3. Prints one row per strategy: mean and 95% bootstrap interval for each metric. Strategies
       whose intervals overlap are not clearly different.
    4. Lists up to 10 failing cases per strategy with their top-5 chunk ids.
    5. Saves a JSON and Markdown report per strategy in `reports/` unless `--no-save` is given.

Examples:
    uv run python scripts/run_retrieval_eval.py
    uv run python scripts/run_retrieval_eval.py --split heldout --strategy hybrid --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Modules import each other as `src.*`, `rag.*`, `core.*`, `datasets.*` and `evaluation.*`.
for p in (ROOT, ROOT / "src", ROOT / "evaluation"):
    sys.path.insert(0, str(p))

from evaluation.core.report import save_report  # noqa: E402
from evaluation.hard_coded_lib.suites.retrieval import compare_strategies, run_retrieval_suite  # noqa: E402

from rag.retriever import STRATEGIES  # noqa: E402

SHOWN = ("hit@5", "recall@5", "precision@5", "mrr", "ndcg@5", "pass_rate")  # table columns


def main() -> None:
    """Parse CLI options, run the retrieval suite for one or all strategies, print and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, help="default: compare all strategies")
    parser.add_argument("--split", choices=("dev", "heldout"), help="default: all cases")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--no-save", action="store_true", help="do not write reports/")
    args = parser.parse_args()

    if args.strategy:
        reports = {
            args.strategy: run_retrieval_suite(strategy=args.strategy, split=args.split, k=args.k)
        }
    else:
        reports = compare_strategies(split=args.split, k=args.k)

    first = next(iter(reports.values()))
    print(f"\nsplit={args.split or 'all'}  k={args.k}  answerable cases={first.n}\n")
    print(f"{'strategy':<8} " + " ".join(f"{m:>22}" for m in SHOWN))
    for name, rep in reports.items():
        cells = []
        for m in SHOWN:
            s = rep.metrics[m]
            cells.append(f"{s.mean:5.3f} [{s.ci_low:5.3f},{s.ci_high:5.3f}]")
        print(f"{name:<8} " + " ".join(f"{c:>22}" for c in cells))

    for name, rep in reports.items():
        failures = rep.failures()
        print(f"\n{name}: {len(failures)} failing case(s) (recall@5 < 1)")
        for c in failures[:10]:
            print(f"  {c.case_id} [{c.category}] top5={c.details.get('retrieved')}")

    if not args.no_save:
        for rep in reports.values():
            print("saved", save_report(rep))


if __name__ == "__main__":
    main()
