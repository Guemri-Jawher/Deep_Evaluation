"""Generic case runner and metric aggregation, shared by every evaluation suite.

A suite (retrieval, RAG answers, agent trajectories, ...) only supplies its cases and a function
that scores one case. This module does the rest in two steps:

    cases --run_cases--> list[CaseResult] --summarize--> SuiteReport

run_cases
    Calls `fn(case)` for each case in a thread pool (`max_workers`, default 2 to stay under free
    LLM rate limits). Results keep the order of `cases`. If `fn` raises, the case becomes a failed
    `CaseResult` with `error="ExcType: message"` instead of crashing the whole run. A line
    `[suite] i/n ok|FAIL case_id` goes to stderr per case, so stdout stays clean for reports.
    `case_id` maps a case to its id for that error record.

summarize
    Turns the per-case results into one `SuiteReport` (suite name, dataset version, config, case
    count, metrics and the raw cases). Metrics are:
      - every score name found in `CaseResult.scores`, averaged over the cases that have it;
      - `pass_rate` (share with `passed`) and `error_rate` (share that crashed), over all cases;
      - with `include_ops`, `latency_p50_s` and `latency_p95_s` and `tokens_per_case`, from
        non-errored cases only (zero latency or missing tokens are ignored).
    Each metric is a `MetricSummary` with mean, a 95% bootstrap interval (`core.stats`) and `n`,
    the number of values behind it. Latency percentiles are single points, so their interval
    equals the value. Metric names are sorted, so reports diff cleanly between runs.

Crashed cases still count as failures in `pass_rate`, but their scores are missing, so check
`error_rate` before trusting any other metric.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from evaluation.core.stats import bootstrap_ci, percentile
from evaluation.core.types import CaseResult, MetricSummary, SuiteReport

T = TypeVar("T")


def run_cases(
    cases: Sequence[T],
    fn: Callable[[T], CaseResult],
    *,
    case_id: Callable[[T], str],
    suite: str,
    max_workers: int = 2,
    progress: bool = True,
) -> list[CaseResult]:
    """Run `fn` on every case. A crashing case becomes a failed CaseResult, never a crashed run."""

    def safe(case: T) -> CaseResult:
        try:
            return fn(case)
        except Exception as exc:
            return CaseResult(
                case_id=case_id(case),
                suite=suite,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    results: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for i, result in enumerate(pool.map(safe, cases), start=1):
            results.append(result)
            if progress:
                mark = "ok " if result.passed else "FAIL"
                print(f"[{suite}] {i}/{len(cases)} {mark} {result.case_id}", file=sys.stderr)
    return results


def summarize(
    results: Sequence[CaseResult],
    *,
    suite: str,
    dataset_version: str,
    config: dict[str, object] | None = None,
    include_ops: bool = True,
) -> SuiteReport:
    """Aggregate per-case scores into metrics with bootstrap confidence intervals."""
    by_metric: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for name, value in r.scores.items():
            by_metric[name].append(value)
    scored = [r for r in results if r.error is None]
    by_metric["pass_rate"] = [1.0 if r.passed else 0.0 for r in results]
    by_metric["error_rate"] = [0.0 if r.error is None else 1.0 for r in results]

    metrics: dict[str, MetricSummary] = {}
    for name, values in sorted(by_metric.items()):
        mean, lo, hi = bootstrap_ci(values)
        metrics[name] = MetricSummary(name=name, mean=mean, ci_low=lo, ci_high=hi, n=len(values))
    if include_ops and scored:
        latencies = [r.latency_s for r in scored if r.latency_s > 0]
        if latencies:
            for q in (50, 95):
                v = percentile(latencies, q)
                metrics[f"latency_p{q}_s"] = MetricSummary(
                    name=f"latency_p{q}_s", mean=v, ci_low=v, ci_high=v, n=len(latencies)
                )
        tokens = [float(r.tokens) for r in scored if r.tokens]
        if tokens:
            mean, lo, hi = bootstrap_ci(tokens)
            metrics["tokens_per_case"] = MetricSummary(
                name="tokens_per_case", mean=mean, ci_low=lo, ci_high=hi, n=len(tokens)
            )
    return SuiteReport(
        suite=suite,
        dataset_version=dataset_version,
        config=dict(config or {}),
        n=len(results),
        metrics=metrics,
        cases=list(results),
    )
