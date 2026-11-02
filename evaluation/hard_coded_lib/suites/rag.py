"""RAG suite: decompose the system so a failure can be localized.

For each answerable question it scores retrieval (recall@5, MRR) and generation (faithfulness,
answer relevance, context relevance - the RAG triad - plus correctness against the reference and
citation validity). Unanswerable questions measure abstention. Every failed case is diagnosed as a
retrieval or a generation problem.
Per case: `pipeline.answer` -> deterministic scores (citations, token F1, retrieval metrics) ->
four LLM judges scoring 1-5 (stored as 0-1) -> `diagnose` -> `CaseResult`. A case passes only if
every check passes; see doc.md in this folder."""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from datasets.utils.loader import DEFAULT_VERSION, load_cases
from datasets.utils.schema import RagCase
from evaluation.core.runner import run_cases, summarize
from evaluation.core.stats import classification_report
from evaluation.core.types import CaseResult, MetricSummary, SuiteReport
from evaluation.hard_coded_lib.metrics.answer import abstained, citation_scores, token_f1
from evaluation.hard_coded_lib.metrics.retrieval import retrieval_scores
from evaluation.judges import make_judge
from evaluation.judges.judge import LLMJudge, Verdict

from observability.langfuse_client import trace_run
from rag.models import extract_citations
from rag.pipeline import RagPipeline, RagResult
from rag.retriever import Retriever, get_retriever

PASS_SCORE = 4  # a judge score of 4 or 5 (on the 1-5 rubric) counts as a pass


class RagJudges:
    """The four LLM judges used on answerable questions, one rubric each.

    `faithfulness` checks the answer against the retrieved context, `answer_relevance` checks it
    against the question, `context_relevance` checks the context against the question, and
    `correctness` checks the answer against the reference answer.
    """

    def __init__(
        self,
        faithfulness: LLMJudge,
        answer_relevance: LLMJudge,
        context_relevance: LLMJudge,
        correctness: LLMJudge,
    ) -> None:
        self.faithfulness = faithfulness
        self.answer_relevance = answer_relevance
        self.context_relevance = context_relevance
        self.correctness = correctness

    @classmethod
    def default(cls, prompt_label: str | None = None) -> RagJudges:
        """Build the four judges with the configured judge model and the rubrics in the registry."""
        return cls(
            *(
                make_judge(r, prompt_label=prompt_label)
                for r in (
                    "judge_faithfulness",
                    "judge_answer_relevance",
                    "judge_context_relevance",
                    "judge_correctness",
                )
            )
        )


def _run_parallel(*calls: Callable[[], Verdict]) -> list[Verdict]:
    """Run independent judge calls at once; each keeps the caller's Langfuse trace context."""
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, call) for call in calls]
        return [f.result() for f in futures]


def diagnose(recall_at_5: float, faithfulness: float, correctness: float) -> str:
    """Localize a failure using the RAG triad logic from the evaluation guide."""
    if recall_at_5 < 1.0:
        return "retrieval: evidence missing from top-5 (fix chunking, embeddings or retriever)"
    if faithfulness < PASS_SCORE:
        return "generation: answer not grounded in good context (fix the prompt)"
    if correctness < PASS_SCORE:
        return "generation: grounded but incorrect/incomplete versus reference"
    return ""


def evaluate_rag_case(
    case: RagCase, result: RagResult, retriever: Retriever, judges: RagJudges | None
) -> CaseResult:
    """Score one pipeline answer and decide whether the case passed.

    Always computed (no LLM): `has_citation`, `citation_validity` (share of cited ids that were
    really retrieved) and `token_f1` against the reference.

    Unanswerable question: passes only if the answer abstains; no retrieval metrics or judges run.

    Answerable question: adds retrieval metrics at k = 1, 3, 5. With judges it also stores the four
    judge scores as 0-1 values and their reasons, then `diagnose` names the failure. The case
    passes when there is no failure and `citation_validity == 1.0`. Without judges it passes when
    `recall@5 == 1.0` and the citations are valid.
    """
    ranked = [r.chunk for r in result.retrieved]
    cite = citation_scores(result.answer, [c.id for c in ranked])
    base: dict[str, float] = {**cite, "token_f1": token_f1(result.answer, case.reference_answer)}
    did_abstain = abstained(result.answer)
    details: dict[str, object] = {
        "answer": result.answer[:600],
        "retrieved": [c.id for c in ranked],
        "abstained": did_abstain,
    }

    if not case.answerable:
        failure = "" if did_abstain else "answered an unanswerable question"
        return CaseResult(
            case_id=case.id,
            suite="rag",
            category=case.category,
            split=case.split,
            scores=base,
            passed=did_abstain,
            details={**details, "failure": failure},
        )

    scores = {**base, **retrieval_scores(ranked, case.evidence, retriever.chunks, ks=(1, 3, 5))}
    failure = ""
    if judges is not None:
        faith, rel, ctx, corr = _run_parallel(
            lambda: judges.faithfulness.score(context=result.context, answer=result.answer),
            lambda: judges.answer_relevance.score(question=case.question, answer=result.answer),
            lambda: judges.context_relevance.score(question=case.question, context=result.context),
            lambda: judges.correctness.score(
                question=case.question, reference=case.reference_answer, answer=result.answer
            ),
        )
        for name, v in (
            ("faithfulness", faith),
            ("answer_relevance", rel),
            ("context_relevance", ctx),
            ("correctness", corr),
        ):
            if v.score is not None:
                scores[name] = v.normalized
            details[f"{name}_reason"] = v.reasoning[:300]
        f = faith.score or 0
        c = corr.score or 0
        failure = diagnose(scores["recall@5"], f, c)
        passed = (not failure) and cite["citation_validity"] == 1.0
    else:
        passed = scores["recall@5"] >= 1.0 and cite["citation_validity"] == 1.0
        failure = "" if passed else "retrieval or citation problem (no judge)"
    return CaseResult(
        case_id=case.id,
        suite="rag",
        category=case.category,
        split=case.split,
        scores=scores,
        passed=passed,
        details={**details, "failure": failure},
    )


def run_rag_suite(
    *,
    pipeline: RagPipeline | None = None,
    judges: RagJudges | None = None,
    split: str | None = None,
    limit: int | None = None,
    max_workers: int = 2,
    version: str = DEFAULT_VERSION,
    use_judges: bool = True,
) -> SuiteReport:
    """Run the pipeline on the golden set and return the aggregated report.

    `use_judges=False` skips the four LLM calls per case (cheap, deterministic checks only).
    `limit` keeps the first N cases of the file; the unanswerable ones are at the end, so a small
    limit gives no `abstain_*` metrics. Cases run in `max_workers` threads.
    """
    retriever = pipeline.retriever if pipeline else get_retriever()
    if pipeline is None:
        from llm.llm import agent_model

        pipeline = RagPipeline(retriever, agent_model())
    if use_judges and judges is None:
        judges = RagJudges.default()
    cases = load_cases("rag_golden", split=split, version=version)
    if limit:
        cases = cases[:limit]

    def run(case: RagCase) -> CaseResult:
        import time

        t0 = time.perf_counter()
        with trace_run(
            f"rag:{case.id}",
            input=case.question,
            tags=["hard-coded", "rag"],
            metadata={"case_id": case.id, "category": case.category},
        ):
            result = pipeline.answer(case.question)
            out = evaluate_rag_case(case, result, retriever, judges if use_judges else None)
        out.latency_s = time.perf_counter() - t0
        return out

    results = run_cases(cases, run, case_id=lambda c: c.id, suite="rag", max_workers=max_workers)
    report = summarize(
        results,
        suite="rag",
        dataset_version=version,
        config={
            "strategy": pipeline.strategy or retriever.default_strategy,
            "k": pipeline.k,
            "judges": use_judges,
            "split": split,
        },
    )
    _add_abstention_metrics(report, cases)
    return report


def _add_abstention_metrics(report: SuiteReport, cases: list[RagCase]) -> None:
    """Treat 'should abstain' as the positive class: precision/recall of the abstain behaviour."""
    by_id = {c.case_id: c for c in report.cases}
    y_true: list[bool] = []
    y_pred: list[bool] = []
    for case in cases:
        res = by_id.get(case.id)
        if res is None or res.error is not None:
            continue
        y_true.append(not case.answerable)
        y_pred.append(bool(res.details.get("abstained", False)))
    if y_true and any(y_true):
        rep = classification_report(y_true, y_pred)
        for key in ("precision", "recall", "f1"):
            report.metrics[f"abstain_{key}"] = MetricSummary(
                name=f"abstain_{key}",
                mean=rep[key],
                ci_low=rep[key],
                ci_high=rep[key],
                n=len(y_true),
            )


__all__ = ["RagJudges", "diagnose", "evaluate_rag_case", "extract_citations", "run_rag_suite"]
