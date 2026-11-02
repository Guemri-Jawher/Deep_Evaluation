"""Persist, render and compare suite reports.

A `SuiteReport` (built by `core.runner.summarize`) is the single artifact of an evaluation run.
This module covers its life after the run: write it to disk, read it back, show it, and check
whether a new run beat an old one.

save_report
    Writes three files to `REPORTS_DIR` (or `directory`) and returns the path of the JSON one:
      - `<timestamp>-<suite>.json`  full report, including every case, for later comparison;
      - `<timestamp>-<suite>.md`    human-readable summary from `render_markdown`;
      - `latest-<suite>.json`       a copy overwritten on each run, so tools and CI can always
                                    load the newest report without knowing its timestamp.
    The timestamp comes from the report's `created_at`, with `:` and `-` stripped so the name
    is valid on Windows.

load_report
    Reads a saved JSON file back into a validated `SuiteReport`.

render_markdown
    A title with the creation time, the dataset version, case count and config, then a table of
    every metric (mean, 95% CI, n), then a failures section. Each failure shows the case id, its
    category and the error, or the `fail*` entries from the case details. At most `max_failures`
    (default 15) are listed, each cut to 200 characters. The section header shows the full count.

compare_reports
    Compares a candidate run with a baseline, metric by metric, returning `delta` (candidate minus
    baseline), `ci_low`, `ci_high` and `significant`. A change is significant when the 95%
    interval excludes 0. It pairs the two runs by case id and uses a paired bootstrap
    (`core.stats.paired_bootstrap_delta`), so it cancels differences in case difficulty and only
    cases present in both runs count. Metrics missing from either report are skipped.

    Limits: with fewer than 2 paired cases, or for metrics that are not stored in the per-case
    `scores` (such as `pass_rate`, `error_rate` and latency or token metrics, which `summarize`
    computes), it falls back to the difference of the two means with no real interval. There
    `ci_low == ci_high == delta`, so `significant` is true for any nonzero difference. Treat
    `significant` as meaningful only for per-case score metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.core.stats import paired_bootstrap_delta
from evaluation.core.types import SuiteReport
from src.utils.config import REPORTS_DIR


def save_report(report: SuiteReport, directory: Path | None = None) -> Path:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.replace(":", "").replace("-", "")[:15]
    base = directory / f"{stamp}-{report.suite}"
    base.with_suffix(".json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    (directory / f"latest-{report.suite}.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return base.with_suffix(".json")


def load_report(path: Path) -> SuiteReport:
    return SuiteReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def render_markdown(report: SuiteReport, *, max_failures: int = 15) -> str:
    lines = [
        f"# {report.suite} ({report.created_at})",
        "",
        f"- dataset: {report.dataset_version}, cases: {report.n}",
        f"- config: `{json.dumps(report.config, default=str)}`",
        "",
        "| metric | mean | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for m in report.metrics.values():
        lines.append(f"| {m.name} | {m.mean:.3f} | [{m.ci_low:.3f}, {m.ci_high:.3f}] | {m.n} |")
    failures = report.failures()
    lines += ["", f"## Failures ({len(failures)})", ""]
    for c in failures[:max_failures]:
        why = c.error or "; ".join(f"{k}={v}" for k, v in c.details.items() if k.startswith("fail"))
        lines.append(f"- `{c.case_id}` [{c.category}] {why[:200]}")
    return "\n".join(lines) + "\n"


def compare_reports(
    baseline: SuiteReport, candidate: SuiteReport
) -> dict[str, dict[str, float | bool]]:
    """Per-metric delta (candidate - baseline) using paired bootstrap on shared case ids."""
    base_cases = {c.case_id: c for c in baseline.cases}
    result: dict[str, dict[str, float | bool]] = {}
    for name in candidate.metrics:
        if name not in baseline.metrics:
            continue
        pairs = [
            (base_cases[c.case_id].scores[name], c.scores[name])
            for c in candidate.cases
            if c.case_id in base_cases and name in c.scores and name in base_cases[c.case_id].scores
        ]
        if len(pairs) >= 2:
            delta, lo, hi = paired_bootstrap_delta([p[0] for p in pairs], [p[1] for p in pairs])
        else:
            delta = candidate.metrics[name].mean - baseline.metrics[name].mean
            lo = hi = delta
        result[name] = {
            "delta": delta,
            "ci_low": lo,
            "ci_high": hi,
            "significant": lo > 0 or hi < 0,
        }
    return result
