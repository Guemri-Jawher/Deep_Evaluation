"""DeepEval-backed suites that emit the same SuiteReport as the native suites (so gate/reports/compare work)."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

from datasets.utils.loader import DEFAULT_VERSION, load_cases
from datasets.utils.schema import RagCase
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from evaluation.core.runner import run_cases, summarize
from evaluation.core.types import CaseResult, SuiteReport
from evaluation.deep_eval.model import LangChainJudge
from evaluation.deep_eval.rag_eval.case import (
    RAG_METRICS,
    RETRIEVAL_METRICS,
    THRESHOLD,
    MetricFactory,
    rag_metric_names,
    rag_test_case,
    retrieval_test_case,
)
from langchain_core.language_models.chat_models import BaseChatModel

from observability.langfuse_client import trace_run
from rag.pipeline import RagPipeline
from rag.retriever import Retriever, get_retriever

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # keep eval data local
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")


def measure(
    test_case: LLMTestCase, factories: Mapping[str, MetricFactory], judge: LangChainJudge
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Run each metric on a fresh instance (metrics keep per-case state)."""
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    failed: list[str] = []
    for name, factory in factories.items():
        metric: BaseMetric = factory(judge)
        metric.measure(test_case)
        scores[f"de_{name}"] = float(metric.score or 0.0)
        reasons[name] = str(getattr(metric, "reason", "") or "")
        if not metric.is_successful():
            failed.append(name)
    return scores, reasons, failed


def run_deepeval_retrieval_suite(
    judge_llm: BaseChatModel,
    *,
    retriever: Retriever | None = None,
    strategy: str | None = None,
    k: int = 5,
    split: str | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    version: str = DEFAULT_VERSION,
) -> SuiteReport:
    """LLM-judged context quality (DeepEval contextual relevancy, precision and recall) of the top-k chunks."""
    retriever = retriever or get_retriever()
    strategy = strategy or retriever.default_strategy
    judge = LangChainJudge(judge_llm)
    cases = [c for c in load_cases("rag_golden", split=split, version=version) if c.answerable]
    if limit:
        cases = cases[:limit]

    def run(case: RagCase) -> CaseResult:
        t0 = time.perf_counter()
        with trace_run(
            f"deepeval-retrieval:{case.id}",
            input=case.question,
            tags=["deepeval", "retrieval"],
            metadata={"case_id": case.id, "strategy": strategy},
        ):
            chunks = [h.chunk for h in retriever.retrieve(case.question, k=k, strategy=strategy)]
            scores, reasons, failed = measure(
                retrieval_test_case(case, chunks),
                {n: RAG_METRICS[n] for n in RETRIEVAL_METRICS},
                judge,
            )
        return CaseResult(
            case_id=case.id,
            suite=f"deepeval-retrieval-{strategy}",
            category=case.category,
            split=case.split,
            scores=scores,
            passed=not failed,
            details={
                "retrieved": [c.id for c in chunks],
                "failed_metrics": failed,
                "reasons": reasons,
            },
            latency_s=time.perf_counter() - t0,
        )

    suite = f"deepeval-retrieval-{strategy}"
    results = run_cases(cases, run, case_id=lambda c: c.id, suite=suite, max_workers=max_workers)
    return summarize(
        results,
        suite=suite,
        dataset_version=version,
        config={"strategy": strategy, "k": k, "split": split, "judge": judge.get_model_name()},
    )


def run_deepeval_rag_suite(
    judge_llm: BaseChatModel,
    *,
    pipeline: RagPipeline | None = None,
    split: str | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    version: str = DEFAULT_VERSION,
) -> SuiteReport:
    if pipeline is None:
        from llm.llm import agent_model

        pipeline = RagPipeline(get_retriever(), agent_model())
    judge = LangChainJudge(judge_llm)
    cases: list[RagCase] = load_cases("rag_golden", split=split, version=version)
    if limit:
        cases = cases[:limit]

    def run(case: RagCase) -> CaseResult:
        t0 = time.perf_counter()
        with trace_run(
            f"deepeval-rag:{case.id}",
            input=case.question,
            tags=["deepeval", "rag"],
            metadata={"case_id": case.id, "category": case.category},
        ):
            result = pipeline.answer(case.question)
            names = rag_metric_names(case)
            scores, reasons, failed = measure(
                rag_test_case(case, result), {n: RAG_METRICS[n] for n in names}, judge
            )
        return CaseResult(
            case_id=case.id,
            suite="deepeval-rag",
            category=case.category,
            split=case.split,
            scores=scores,
            passed=not failed,
            details={"answer": result.answer[:500], "failed_metrics": failed, "reasons": reasons},
            latency_s=time.perf_counter() - t0,
        )

    results = run_cases(
        cases, run, case_id=lambda c: c.id, suite="deepeval-rag", max_workers=max_workers
    )
    return summarize(
        results,
        suite="deepeval-rag",
        dataset_version=version,
        config={"split": split, "judge": judge.get_model_name(), "threshold": THRESHOLD},
    )
