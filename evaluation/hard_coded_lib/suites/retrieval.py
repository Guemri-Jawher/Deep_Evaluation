"""Retrieval suite (no LLM): recall@k, precision@k, MRR, NDCG, hit rate per strategy.

This is the first thing to look at when a RAG answer is wrong: if recall@5 is low, prompt tuning
cannot fix it (evaluation guide, section 4).
"""

from __future__ import annotations

from datasets.utils.loader import DEFAULT_VERSION, load_cases
from datasets.utils.schema import RagCase
from evaluation.core.runner import run_cases, summarize
from evaluation.core.types import CaseResult, SuiteReport
from evaluation.hard_coded_lib.metrics.retrieval import retrieval_scores

from rag.retriever import STRATEGIES, Retriever, get_retriever


def run_retrieval_suite(
    *,
    strategy: str | None = None,
    split: str | None = None,
    k: int = 10,
    retriever: Retriever | None = None,
    cases: list[RagCase] | None = None,
    version: str = DEFAULT_VERSION,
) -> SuiteReport:
    """
    Run the retrieval suite for a given strategy, dataset split, and top-k setting.

    Returns a `SuiteReport` summarizing the retrieval performance.
    Args:
        strategy: The retrieval strategy to use. If None, the retriever's default strategy is used.
        split: The dataset split to evaluate on.
        k: The number of top documents to retrieve.
        retriever: An optional Retriever instance. If None, a default retriever is obtained.
        cases: Optional list of RagCase instances to evaluate. If None, cases are loaded from the dataset.
        version: The dataset version to use.

    Returns:
        A `SuiteReport` object containing the summarized retrieval performance.
    """
    retriever = retriever or get_retriever()
    strategy = strategy or retriever.default_strategy
    cases = cases if cases is not None else load_cases("rag_golden", split=split, version=version)
    answerable = [c for c in cases if c.answerable]

    def evaluate(case: RagCase) -> CaseResult:
        hits = retriever.retrieve(case.question, k=k, strategy=strategy)
        ranked = [h.chunk for h in hits]
        scores = retrieval_scores(ranked, case.evidence, retriever.chunks)
        return CaseResult(
            case_id=case.id,
            suite=f"retrieval-{strategy}",
            category=case.category,
            split=case.split,
            scores=scores,
            passed=scores["recall@5"] >= 1.0,
            details={
                "retrieved": [c.id for c in ranked[:5]],
                "fail": "" if scores["recall@5"] >= 1.0 else "evidence not in top 5",
            },
        )

    results = run_cases(
        answerable,
        evaluate,
        case_id=lambda c: c.id,
        suite=f"retrieval-{strategy}",
        max_workers=1,
        progress=False,
    )
    return summarize(
        results,
        suite=f"retrieval-{strategy}",
        dataset_version=version,
        include_ops=False,
        config={"strategy": strategy, "split": split, "k": k, "chunks": len(retriever.chunks)},
    )


def compare_strategies(**kwargs: object) -> dict[str, SuiteReport]:
    """Baseline vs candidate retrieval strategies on the same fixed dataset."""
    return {s: run_retrieval_suite(strategy=s, **kwargs) for s in STRATEGIES}  # type: ignore[arg-type]
