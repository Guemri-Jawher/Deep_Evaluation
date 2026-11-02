"""BM25 over rank_bm25 with a tokenizer that keeps `pass^k` / `recall@5`, plus reciprocal-rank fusion."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[@^][a-z0-9]+)?")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "with",
        "do",
        "does",
        "did",
        "can",
        "should",
        "would",
        "will",
        "your",
        "you",
        "we",
        "they",
        "their",
        "there",
        "these",
        "those",
        "into",
        "than",
        "then",
        "so",
        "if",
        "not",
        "no",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Client:
    """Score-aware BM25 search over documents, with optional filtering by `source` metadata."""

    def __init__(self, documents: list[Document]) -> None:
        self._docs = documents
        # BM25Okapi cannot be built from an empty corpus (schema-only retrievers in tests/evals).
        self._index = (
            BM25Okapi([tokenize(d.page_content) for d in documents]) if documents else None
        )

    def search(
        self, query: str, k: int = 4, source: str | None = None
    ) -> list[tuple[Document, float]]:
        """Top-`k` (document, BM25 score) pairs; zero-score documents and other sources are excluded."""
        if self._index is None:
            return []
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(zip(self._docs, scores, strict=True), key=lambda p: p[1], reverse=True)
        hits = [
            (doc, float(score))
            for doc, score in ranked
            if score > 0 and (source is None or doc.metadata.get("source") == source)
        ]
        return hits[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several ranked id lists into one score per id."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
