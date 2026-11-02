"""Map our result objects onto DeepEval test cases and pick the metrics for each case type."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from datasets.utils.schema import RagCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rag.models import Chunk
from rag.pipeline import RagResult

THRESHOLD = 0.7
MetricFactory = Callable[[DeepEvalBaseLLM], BaseMetric]


def rag_metric_names(case: RagCase) -> tuple[str, ...]:
    """Metrics for the full pipeline: all of `RAG_METRICS` if answerable, else `RAG_METRICS_UNANSWERABLE`."""
    return tuple(RAG_METRICS) if case.answerable else RAG_METRICS_UNANSWERABLE


def retrieval_test_case(case: RagCase, chunks: Sequence[Chunk]) -> LLMTestCase:
    # No generator here: the reference answer stands in as actual_output so only the context is judged.
    return LLMTestCase(
        input=case.question,
        actual_output=case.reference_answer,
        expected_output=case.reference_answer,
        retrieval_context=[f"[{c.id}] {c.text}" for c in chunks],
        name=case.id,
    )


def rag_test_case(case: RagCase, result: RagResult) -> LLMTestCase:
    return LLMTestCase(
        input=case.question,
        actual_output=result.answer,
        expected_output=case.reference_answer,
        retrieval_context=[f"[{r.chunk.id}] {r.chunk.text}" for r in result.retrieved],
        name=case.id,
    )


def _simple(cls: type[Any]) -> MetricFactory:
    def make(model: DeepEvalBaseLLM) -> BaseMetric:
        metric: BaseMetric = cls(model=model, threshold=THRESHOLD, async_mode=True)
        return metric

    return make


def _correctness(model: DeepEvalBaseLLM) -> BaseMetric:
    """
    Correctness metric: checks if the actual output is factually consistent with the expected output.

    Args:
        model: The language model to use for evaluation.
    Returns:
        An instance of the correctness metric.
    """
    return GEval(
        name="Correctness",
        criteria=(
            "Is the actual output factually consistent with the expected output? Penalize missing "
            "key facts and contradictions; do not penalize different wording or extra citations."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=THRESHOLD,
        async_mode=True,
    )


# RAG triad (faithfulness, answer relevancy, context relevancy) plus retrieval ranking/recall.
RAG_METRICS: dict[str, MetricFactory] = {
    "faithfulness": _simple(FaithfulnessMetric),
    "answer_relevancy": _simple(AnswerRelevancyMetric),
    "contextual_relevancy": _simple(ContextualRelevancyMetric),
    "contextual_precision": _simple(ContextualPrecisionMetric),
    "contextual_recall": _simple(ContextualRecallMetric),
    "correctness": _correctness,
}
# An unanswerable question has no gold evidence: only grounding of the abstention matters.
RAG_METRICS_UNANSWERABLE = ("faithfulness", "answer_relevancy")

RETRIEVAL_METRICS = ("contextual_relevancy", "contextual_precision", "contextual_recall")
