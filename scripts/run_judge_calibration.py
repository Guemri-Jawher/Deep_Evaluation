"""Calibrate the LLM judges against human labels and probe them for bias.

For each rubric (faithfulness, correctness, answer_relevance) the judge scores every labelled
example and is compared with the human score. The pairwise judge is then probed for position and
verbosity bias. Cost: 1 call per label, 2 per pairwise probe in each probe; replies are cached on
disk, so a rerun of the same inputs is free. How to read the numbers: evaluation/judges/doc.md.

Examples:
    uv run python scripts/run_judge_calibration.py --rubric faithfulness --limit 5 --no-probes
    uv run python scripts/run_judge_calibration.py --workers 4
"""

from __future__ import annotations

import argparse
import contextvars
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # judge text has characters cp1252 lacks

from datasets.utils.loader import load_cases  # noqa: E402
from datasets.utils.schema import JudgeLabel  # noqa: E402
from evaluation.judges import make_judge  # noqa: E402
from evaluation.judges.calibration import (  # noqa: E402
    CalibrationReport,
    calibrate,
    probe_position_bias,
    probe_verbosity_bias,
)
from evaluation.judges.judge import LLMJudge, Verdict  # noqa: E402

from observability.langfuse_client import flush, trace_run  # noqa: E402
from utils.config import REPORTS_DIR, get_settings  # noqa: E402

RUBRICS = ("faithfulness", "correctness", "answer_relevance")
MIN_RELIABLE_N = 20  # below this, kappa is too noisy to quote


def _score_all(judge: LLMJudge, labels: list[JudgeLabel], workers: int) -> dict[str, Verdict]:
    """Score every label. A judge call that raises becomes an unscored verdict, not a crash."""

    def one(label: JudgeLabel) -> Verdict:
        try:
            return judge.score(**label.inputs)
        except Exception as exc:
            return Verdict(None, f"error: {type(exc).__name__}: {exc}")

    if workers <= 1:
        return {lab.id: one(lab) for lab in labels}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {lab.id: pool.submit(contextvars.copy_context().run, one, lab) for lab in labels}
        return {key: fut.result() for key, fut in futures.items()}


def run_rubric(
    rubric: str, labels: list[JudgeLabel], workers: int, use_cache: bool
) -> dict[str, Any]:
    """Calibrate one rubric; also return the labels where judge and human differ by 2 or more."""
    judge = make_judge(f"judge_{rubric}", use_cache=use_cache)
    with trace_run(
        f"judge-calibration:{rubric}", tags=["judge", "calibration"], metadata={"rubric": rubric}
    ):
        verdicts = _score_all(judge, labels, workers)
    report = calibrate(labels, lambda lab: verdicts[lab.id].score)
    disagreements = [
        {
            "id": lab.id,
            "human": lab.human_score,
            "judge": verdicts[lab.id].score,
            "reasoning": verdicts[lab.id].reasoning[:300],
        }
        for lab in labels
        if verdicts[lab.id].score is not None and abs(verdicts[lab.id].score - lab.human_score) >= 2  # type: ignore[operator]
    ]
    errors = [
        {"id": key, "error": v.reasoning[:200]}
        for key, v in verdicts.items()
        if v.reasoning.startswith("error:")
    ]
    return {"report": asdict(report), "disagreements": disagreements, "errors": errors}


def run_probes(limit: int | None, use_cache: bool) -> dict[str, Any]:
    """Probe the pairwise judge for position and verbosity bias."""
    probes = load_cases("pairwise_probes")
    if limit:
        probes = probes[:limit]
    judge = make_judge("judge_pairwise", use_cache=use_cache)
    out: dict[str, Any] = {"n_probes": len(probes)}
    with trace_run("judge-probes", tags=["judge", "probes"]):
        for name, probe in (
            ("position", probe_position_bias),
            ("verbosity", probe_verbosity_bias),
        ):
            try:
                out[name] = probe(judge, probes)
            except Exception as exc:
                out[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def _num(x: float) -> str:
    return "  n/a" if isinstance(x, float) and math.isnan(x) else f"{x:5.2f}"


def show(results: dict[str, Any]) -> None:
    """Print the calibration table, the disagreements and the probe results."""
    rubrics = results["rubrics"]
    if rubrics:
        print(f"\n=== calibration  judge={results['judge_model']} ({results['provider']})")
        print(
            f"{'rubric':<17} {'n':>3} {'parse':>6} {'exact':>6} {'w1':>6} "
            f"{'wkappa':>6} {'spear':>6} {'passK':>6} {'bias':>6}"
        )
    for rubric, data in rubrics.items():
        r = data["report"]
        print(
            f"{rubric:<17} {r['n']:>3} {_num(r['parse_rate']):>6} {_num(r['exact']):>6} "
            f"{_num(r['within_1']):>6} {_num(r['weighted_kappa']):>6} {_num(r['spearman']):>6} "
            f"{_num(r['pass_kappa']):>6} {_num(r['mean_bias']):>6}"
        )
        if r["n"] < MIN_RELIABLE_N:
            print(f"   note: n={r['n']} is small, treat these numbers as indicative only")
        for err in data["errors"][:3]:
            print(f"   error on {err['id']}: {err['error']}")
    for rubric, data in rubrics.items():
        for d in data["disagreements"]:
            print(
                f"\n[{rubric}] {d['id']}: human={d['human']} judge={d['judge']}\n   {d['reasoning']}"
            )
    if "probes" in results:
        print("\n=== bias probes")
        for name in ("position", "verbosity"):
            print(f"  {name}: {results['probes'].get(name)}")


def _clean(value: Any) -> Any:
    """Replace NaN with None so the JSON stays valid."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def save(results: dict[str, Any]) -> Path:
    """Write the results as JSON and a short Markdown table in reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    base = REPORTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-judge-calibration"
    base.with_suffix(".json").write_text(
        json.dumps(_clean(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# Judge calibration ({results['created_at']})",
        "",
        f"- judge: {results['judge_model']} ({results['provider']})",
        "",
        "| rubric | n | parse_rate | exact | within_1 | weighted_kappa | spearman | pass_kappa | mean_bias |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rubric, data in results["rubrics"].items():
        r = data["report"]
        cells = [_num(r[k]).strip() for k in CalibrationReport.__dataclass_fields__ if k != "n"]
        lines.append(f"| {rubric} | {r['n']} | " + " | ".join(cells) + " |")
    if "probes" in results:
        lines += ["", "## Bias probes", "", f"`{json.dumps(_clean(results['probes']))}`"]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".json")


def main() -> None:
    """Parse options, run the calibrations and probes, print and save the results."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--rubric", choices=(*RUBRICS, "all"), default="all")
    parser.add_argument("--limit", type=int, help="labels per rubric and probes to use")
    parser.add_argument("--no-probes", action="store_true", help="skip the bias probes")
    parser.add_argument("--probes-only", action="store_true", help="skip the calibration")
    parser.add_argument("--workers", type=int, default=1, help="parallel judge calls")
    parser.add_argument("--no-cache", action="store_true", help="ignore the judge reply cache")
    parser.add_argument("--no-save", action="store_true", help="do not write reports/")
    args = parser.parse_args()

    s = get_settings()
    results: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": s.llm_provider,
        "judge_model": s.model_for("judge"),
        "rubrics": {},
    }
    if not args.probes_only:
        labels = load_cases("judge_labels")
        for rubric in RUBRICS if args.rubric == "all" else (args.rubric,):
            subset = [lab for lab in labels if lab.rubric == rubric]
            if args.limit:
                subset = subset[: args.limit]
            print(f"calibrating {rubric}: {len(subset)} labels", file=sys.stderr, flush=True)
            results["rubrics"][rubric] = run_rubric(rubric, subset, args.workers, not args.no_cache)
    if not args.no_probes:
        print("running bias probes", file=sys.stderr, flush=True)
        results["probes"] = run_probes(args.limit, not args.no_cache)

    show(results)
    if not args.no_save:
        print("saved", save(results))


if __name__ == "__main__":
    try:
        main()
    finally:
        flush()  # a short-lived process must push pending traces before exiting
