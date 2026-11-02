"""Judges are systems that must themselves be evaluated.

* `calibrate` measures agreement with human labels (exact, within-1, quadratic kappa, Spearman,
  and binary pass/fail kappa) - the number to quote when you say "we trust this judge".
* `probe_position_bias` and `probe_verbosity_bias` measure the two most common judge biases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from datasets.utils.schema import JudgeLabel, PairwiseProbe
from evaluation.core.stats import cohens_kappa, spearman, weighted_kappa
from evaluation.judges.judge import PASS_THRESHOLD, LLMJudge, compare_pair

ScoreFn = Callable[[JudgeLabel], int | None]


@dataclass
class CalibrationReport:
    n: int
    parse_rate: float
    exact: float
    within_1: float
    weighted_kappa: float
    spearman: float
    pass_kappa: float  # agreement on the pass/fail decision (>= 4), the one used for gating
    mean_bias: float  # judge - human; > 0 means lenient

    def as_metrics(self) -> dict[str, float]:
        return {
            "judge_parse_rate": self.parse_rate,
            "judge_exact": self.exact,
            "judge_within_1": self.within_1,
            "judge_weighted_kappa": self.weighted_kappa,
            "judge_spearman": self.spearman,
            "judge_pass_kappa": self.pass_kappa,
            "judge_mean_bias": self.mean_bias,
        }


def calibrate(labels: Sequence[JudgeLabel], score_fn: ScoreFn) -> CalibrationReport:
    """Compare the judge's scores with human scores on labelled examples.

    `score_fn` returns the judge's 1-5 score for a label, or None if it failed to parse; failures
    lower `parse_rate` and are left out of the agreement metrics. With fewer than 2 scored examples
    the agreement metrics are NaN.
    """
    pairs: list[tuple[int, int]] = []
    for label in labels:
        judged = score_fn(label)
        if judged is not None:
            pairs.append((label.human_score, judged))
    n_ok = len(pairs)
    if n_ok < 2:
        nan = float("nan")
        return CalibrationReport(
            len(labels), n_ok / max(1, len(labels)), nan, nan, nan, nan, nan, nan
        )
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    return CalibrationReport(
        n=len(labels),
        parse_rate=n_ok / len(labels),
        exact=sum(h == j for h, j in pairs) / n_ok,
        within_1=sum(abs(h - j) <= 1 for h, j in pairs) / n_ok,
        weighted_kappa=weighted_kappa(human, judge, levels=[1, 2, 3, 4, 5]),
        spearman=spearman(human, judge),
        pass_kappa=cohens_kappa(
            [h >= PASS_THRESHOLD for h in human], [j >= PASS_THRESHOLD for j in judge]
        ),
        mean_bias=sum(j - h for h, j in pairs) / n_ok,
    )


def probe_position_bias(judge: LLMJudge, probes: Sequence[PairwiseProbe]) -> dict[str, float]:
    """Judge each pair in both orders. Without position bias the verdict flips consistently."""
    consistent = 0
    first_wins = 0
    total = 0
    for p in probes:
        v1 = judge.score_pairwise(
            question=p.question, reference=p.reference, answer_a=p.good, answer_b=p.bad
        )
        v2 = judge.score_pairwise(
            question=p.question, reference=p.reference, answer_a=p.bad, answer_b=p.good
        )
        if v1.winner is None or v2.winner is None:
            continue
        total += 1
        first_wins += (v1.winner == "A") + (v2.winner == "A")
        good_first = {"A": "good", "B": "bad", "tie": "tie"}[v1.winner]
        good_second = {"A": "bad", "B": "good", "tie": "tie"}[v2.winner]
        consistent += good_first == good_second
    if not total:
        return {"position_consistency": float("nan"), "first_position_win_rate": float("nan")}
    return {
        "position_consistency": consistent / total,
        "first_position_win_rate": first_wins / (2 * total),  # 0.5 = no position preference
    }


def probe_verbosity_bias(judge: LLMJudge, probes: Sequence[PairwiseProbe]) -> dict[str, float]:
    """On 'verbosity' probes the concise answer is correct and the long one is padded or wrong."""
    wins = 0
    total = 0
    for p in (p for p in probes if p.kind == "verbosity"):
        result = compare_pair(
            judge, question=p.question, reference=p.reference, first=p.good, second=p.bad
        )
        total += 1
        wins += result.winner == "first"
    return {"prefers_concise_correct": wins / total if total else float("nan"), "n": float(total)}
