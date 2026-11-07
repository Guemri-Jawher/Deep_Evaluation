"""Judge validation suite: prove an LLM judge agrees with humans before it is allowed to gate anything.

Checks the judge is trustworthy before its scores are used.

Reports agreement with human labels per rubric (exact, within-1, weighted kappa, pass/fail kappa)
and two bias probes: position bias (order swap) and verbosity bias (padded wrong vs concise right).
"""

from __future__ import annotations

from collections.abc import Callable

from core.runner import summarize
from core.types import CaseResult, MetricSummary, SuiteReport
from datasets.utils.loader import DEFAULT_VERSION, load_cases
from datasets.utils.schema import JudgeLabel, PairwiseProbe
from judges import make_judge
from judges.calibration import calibrate, probe_position_bias, probe_verbosity_bias
from judges.judge import LLMJudge
from langchain_core.language_models.chat_models import BaseChatModel

RUBRIC_PROMPTS = {
    "faithfulness": "judge_faithfulness",
    "correctness": "judge_correctness",
    "answer_relevance": "judge_answer_relevance",
}


def _recording_scorer(
    judge: LLMJudge, judged: dict[str, int | None]
) -> Callable[[JudgeLabel], int | None]:
    def score(label: JudgeLabel) -> int | None:
        value: int | None = judge.score(**label.inputs).score
        judged[label.id] = value
        return value

    return score


def run_judge_validation(
    *,
    llm: BaseChatModel | None = None,
    rubrics: tuple[str, ...] = ("faithfulness", "correctness", "answer_relevance"),
    prompt_label: str | None = None,
    use_cache: bool = True,
    version: str = DEFAULT_VERSION,
    model_name: str = "judge",
) -> SuiteReport:
    labels: list[JudgeLabel] = load_cases("judge_labels", version=version)
    probes: list[PairwiseProbe] = load_cases("pairwise_probes", version=version)
    cases: list[CaseResult] = []
    extra: dict[str, float] = {}

    for rubric in rubrics:
        judge = make_judge(
            RUBRIC_PROMPTS[rubric], llm=llm, prompt_label=prompt_label, use_cache=use_cache
        )
        rubric_labels = [lb for lb in labels if lb.rubric == rubric]
        judged: dict[str, int | None] = {}
        report = calibrate(rubric_labels, _recording_scorer(judge, judged))
        for name, value in report.as_metrics().items():
            extra[f"{rubric}.{name}"] = value
        for label in rubric_labels:
            j = judged.get(label.id)
            cases.append(
                CaseResult(
                    case_id=label.id,
                    suite="judge",
                    category=rubric,
                    scores={
                        "abs_error": float(abs(j - label.human_score)) if j is not None else 4.0
                    },
                    passed=j is not None and abs(j - label.human_score) <= 1,
                    details={
                        "human": label.human_score,
                        "judge": j,
                        "failure": ""
                        if j is not None and abs(j - label.human_score) <= 1
                        else f"human={label.human_score} judge={j}",
                    },
                )
            )

    pair_judge = make_judge(
        "judge_pairwise", llm=llm, prompt_label=prompt_label, use_cache=use_cache
    )
    extra.update(probe_position_bias(pair_judge, probes))
    verbosity = probe_verbosity_bias(pair_judge, probes)
    extra["prefers_concise_correct"] = verbosity["prefers_concise_correct"]

    out = summarize(
        cases,
        suite="judge",
        dataset_version=version,
        include_ops=False,
        config={"judge_model": model_name, "rubrics": list(rubrics), "n_labels": len(labels)},
    )
    for name, value in extra.items():
        out.metrics[name] = MetricSummary(
            name=name, mean=value, ci_low=value, ci_high=value, n=len(labels)
        )
    return out
