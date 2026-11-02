"""Retrieval metrics: did we fetch the right context?

Relevance is defined by evidence units rather than chunk ids, so labels survive re-chunking:
a chunk is relevant to a unit if it is on one of the unit's pages of its source and contains one
of the unit's phrases (case and whitespace normalized).

Conventions used by every metric below:
    ranked    retrieved chunks, best first (rank 1 is index 0).
    evidence  gold `Evidence` units of one question; a multi-hop question has several.
    k         cutoff; only `ranked[:k]` is considered.

Questions without evidence (unanswerable) cannot be scored for retrieval: `recall_at_k` returns
NaN for them and the other metrics return 0.0, so callers should skip them when averaging.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from datasets.utils.schema import Evidence
from src.rag.models import Chunk

DEFAULT_KS = (1, 3, 5, 10)


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs (including PDF line breaks) to single spaces."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def matches(chunk: Chunk, unit: Evidence) -> bool:
    """True if the chunk is from the unit's source, on one of its pages, and has any of its phrases.

    Phrases are alternatives: one hit is enough, so a generic phrase makes the whole page match.
    """
    if chunk.source != unit.source or chunk.page not in unit.pages:
        return False
    body = normalize(chunk.text)
    return any(normalize(p) in body for p in unit.phrases)


def relevant_chunks(chunks: Sequence[Chunk], evidence: Sequence[Evidence]) -> list[Chunk]:
    """All chunks matching at least one evidence unit, in input order.

    Run on the full corpus, its length is the `total_relevant` that `ndcg_at_k` needs.
    """
    return [c for c in chunks if any(matches(c, u) for u in evidence)]


def validate_evidence(corpus: Sequence[Chunk], evidence: Sequence[Evidence]) -> list[str]:
    """Describe each evidence unit that matches no corpus chunk (label rot or a mistyped phrase).

    An empty list means every unit is reachable; run it on the dataset before evaluating, since
    an unmatched unit can never be retrieved and silently caps recall.
    """
    return [
        f"no chunk matches {u.source} p{u.pages} {u.phrases}"
        for u in evidence
        if not any(matches(c, u) for c in corpus)
    ]


def hit_rate_at_k(ranked: Sequence[Chunk], evidence: Sequence[Evidence], k: int) -> float:
    """1.0 if any top-k chunk matches any evidence unit, else 0.0 (a per-query binary health check)."""
    return float(any(matches(c, u) for c in ranked[:k] for u in evidence))


def recall_at_k(ranked: Sequence[Chunk], evidence: Sequence[Evidence], k: int) -> float:
    """Fraction of evidence units covered by at least one top-k chunk, in [0, 1].

    Counts units, not chunks: a two-source multi-hop question scores 0.5 if only one source is
    found. Returns NaN when `evidence` is empty.
    """
    if not evidence:
        return float("nan")
    covered = sum(any(matches(c, u) for c in ranked[:k]) for u in evidence)
    return covered / len(evidence)


def precision_at_k(ranked: Sequence[Chunk], evidence: Sequence[Evidence], k: int) -> float:
    """Fraction of the retrieved top-k chunks that match any evidence unit, in [0, 1].

    Divides by the number actually retrieved (at most k); 0.0 when nothing was retrieved.
    """
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(any(matches(c, u) for u in evidence) for c in top) / len(top)


def reciprocal_rank(ranked: Sequence[Chunk], evidence: Sequence[Evidence]) -> float:
    """1 / rank of the first matching chunk (1-based), or 0.0 if none is retrieved.

    Averaged over questions this is MRR. It has no cutoff: it scans the whole `ranked` list.
    """
    for rank, c in enumerate(ranked, start=1):
        if any(matches(c, u) for u in evidence):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[Chunk], evidence: Sequence[Evidence], k: int, *, total_relevant: int
) -> float:
    """Binary-gain NDCG@k, in [0, 1]; rewards relevant chunks that appear higher in the ranking.

    DCG adds 1 / log2(rank + 1) for each relevant chunk in the top k. It is divided by the ideal
    DCG, which places min(k, `total_relevant`) relevant chunks at the top. `total_relevant` is the
    number of relevant chunks in the whole corpus (see `relevant_chunks`). Returns 0.0 when it is 0.
    """
    gains = [1.0 if any(matches(c, u) for u in evidence) else 0.0 for c in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, total_relevant)))
    return dcg / ideal if ideal else 0.0


def retrieval_scores(
    ranked: Sequence[Chunk],
    evidence: Sequence[Evidence],
    corpus: Sequence[Chunk],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """All retrieval metrics for one query, keyed by name.

    Keys: `mrr`, `hit@k` and `recall@k` for each k in `ks`, plus `precision@5` and `ndcg@5`.
    `corpus` is the full chunk list, used only to count relevant chunks for NDCG.
    """
    scores = {"mrr": reciprocal_rank(ranked, evidence)}
    for k in ks:
        scores[f"hit@{k}"] = hit_rate_at_k(ranked, evidence, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, evidence, k)
    scores["precision@5"] = precision_at_k(ranked, evidence, 5)
    scores["ndcg@5"] = ndcg_at_k(
        ranked, evidence, 5, total_relevant=len(relevant_chunks(corpus, evidence))
    )
    return scores
