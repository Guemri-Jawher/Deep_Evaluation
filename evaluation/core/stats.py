"""Statistics for evaluation: uncertainty, reliability metrics and rater agreement.

The evaluation guide asks for ranges instead of single points, enough samples to see signal, and
judges validated against humans. Everything here is small, dependency-light and unit-tested.

Uncertainty (report a range, not a single score)
    mean                    Mean of a list; NaN when empty.
    percentile              q-th percentile (e.g. p95 latency); NaN when empty.
    bootstrap_ci            (mean, low, high) with a percentile bootstrap interval. Resamples the
                            per-case scores with replacement; fixed seed, so results repeat.
    wilson_interval         Interval for a pass rate (successes out of n). Stays inside [0, 1] and
                            is more reliable than the normal approximation when n is small.
    paired_bootstrap_delta  (mean improvement, low, high) of candidate minus baseline, computed
                            per case on the same dataset. If the interval excludes 0, the change
                            is more than noise. Use it to decide ship or revert.

Agent reliability (n runs of one task, c of them successful)
    pass_at_k               P(at least one of k tries succeeds): what a user gets if they can
                            retry. Unbiased estimator from n runs, so k must not exceed n.
    pass_pow_k              P(all k tries succeed): reliability under repeated use. It falls fast
                            as k grows, so it is the headline production metric. Same n >= k rule.

Judge validation (does an automated judge agree with human labels?)
    accuracy                Share of identical labels.
    cohens_kappa            Agreement on nominal labels (e.g. pass/fail), corrected for chance.
    weighted_kappa          Same for ordinal scores such as a 1-5 rubric; the quadratic weights
                            penalize a 1-vs-5 disagreement more than a 1-vs-2 one.
    spearman                Rank correlation between two score lists; NaN if one is constant.
    classification_report   Precision, recall, F1, accuracy and tp/fp/fn/tn for a binary label,
                            e.g. judging "is this answer faithful?".

Private helpers: `_ranks` (average ranks for ties, used by `spearman`) and
`_check_same_length` (raises ValueError on empty or unequal inputs).

Scales: kappa is 1 for perfect agreement and 0 for chance level. Sampling is seeded, so reruns
give the same intervals.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap confidence interval: (mean, low, high)."""
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1 or np.all(arr == arr[0]):
        return float(arr.mean()), float(arr.min()), float(arr.max())
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than normal approx for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(at least one success in k tries) from c successes in n runs."""
    if k > n:
        raise ValueError("k cannot exceed the number of runs n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Estimator of P(all k tries succeed): the reliability users actually experience."""
    if k > n:
        raise ValueError("k cannot exceed the number of runs n")
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    _check_same_length(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def cohens_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Chance-corrected agreement between two raters on nominal labels."""
    _check_same_length(a, b)
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in ca.keys() | cb.keys()) / n**2
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], *, levels: Sequence[int] | None = None
) -> float:
    """Quadratic-weighted kappa for ordinal scores (e.g. 1-5 rubric scores)."""
    _check_same_length(a, b)
    labels = list(levels) if levels is not None else sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(labels)}
    m = len(labels)
    if m < 2:
        return 1.0
    observed = np.zeros((m, m))
    for x, y in zip(a, b, strict=True):
        observed[idx[x], idx[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[(i - j) ** 2 / (m - 1) ** 2 for j in range(m)] for i in range(m)])
    denom = (weights * expected).sum()
    return 1.0 if denom == 0 else float(1 - (weights * observed).sum() / denom)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    _check_same_length(a, b)
    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def paired_bootstrap_delta(
    baseline: Sequence[float], candidate: Sequence[float], *, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean per-case improvement (candidate - baseline) with a 95% bootstrap interval."""
    _check_same_length(baseline, candidate)
    diffs = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    return bootstrap_ci(diffs.tolist(), n_boot=n_boot, seed=seed)


def classification_report(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Precision / recall / F1 / accuracy for a binary label, plus the confusion counts."""
    _check_same_length(y_true, y_pred)
    tp = sum(t and p for t, p in zip(y_true, y_pred, strict=True))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred, strict=True))
    tn = len(y_true) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(len(arr))
    ranks[order] = np.arange(len(arr), dtype=float)
    # average ranks for ties
    for v in np.unique(arr):
        mask = arr == v
        ranks[mask] = ranks[mask].mean()
    return ranks


def _check_same_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b) or not len(a):
        raise ValueError("sequences must be non-empty and the same length")
