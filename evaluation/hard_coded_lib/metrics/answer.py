"""Deterministic answer-level metrics: cheap, rigorous, and immune to judge bias."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from rag.models import extract_citations

_ABSTAIN = re.compile(
    r"(?:guides?\s+(?:do|does)\s+not\s+(?:cover|contain|mention|address)|not\s+covered\s+(?:in|by)\s+the\s+guides?"
    r"|(?:cannot|can't|could\s+not|couldn't)\s+find\s+(?:this|that|it|any)\b[^.]{0,40}\bguides?)",
    re.I,
)
_TOKEN = re.compile(r"[a-z0-9]+")


def abstained(answer: str) -> bool:
    """True if the answer declines, e.g. "The guides do not cover this." (regex on a few phrasings)."""
    return bool(_ABSTAIN.search(answer))


def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric words; punctuation is dropped."""
    return _TOKEN.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words F1. A coarse regression tripwire only (see guide: overlap metrics are weak)."""
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if both texts have the same words in the same order (case and punctuation ignored)."""
    return float(tokens(prediction) == tokens(reference))


def citation_scores(answer: str, retrieved_ids: Iterable[str]) -> dict[str, float]:
    """has_citation: answer cites at least one id. citation_validity: share of cited ids that were
    actually retrieved (1.0 = no fabricated citations)."""
    cited = extract_citations(answer)
    retrieved = set(retrieved_ids)
    return {
        "has_citation": float(bool(cited)),
        "citation_validity": (sum(c in retrieved for c in cited) / len(cited)) if cited else 1.0,
    }


def matches_all(text: str, patterns: Sequence[str]) -> list[str]:
    """Return the patterns that do NOT match (empty list = all matched)."""
    return [p for p in patterns if not re.search(p, text, re.I | re.S)]


def matches_any(text: str, patterns: Sequence[str]) -> bool:
    """True if at least one regex matches, or if there are no patterns."""
    return not patterns or any(re.search(p, text, re.I | re.S) for p in patterns)


def forbidden_hits(text: str, patterns: Sequence[str]) -> list[str]:
    """Return the regexes that DO match; used for text that must not appear."""
    return [p for p in patterns if re.search(p, text, re.I | re.S)]
