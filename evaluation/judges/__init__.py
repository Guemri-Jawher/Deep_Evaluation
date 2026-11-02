"""LLM judges, bias probes and calibration against human labels."""

from __future__ import annotations

from evaluation.judges.judge import LLMJudge, disk_cache
from langchain_core.language_models.chat_models import BaseChatModel

from utils.config import get_settings

RUBRICS = (
    "judge_faithfulness",
    "judge_answer_relevance",
    "judge_context_relevance",
    "judge_correctness",
    "judge_trajectory",
    "judge_pairwise",
    "judge_safety",
)


def make_judge(
    rubric: str,
    *,
    llm: BaseChatModel | None = None,
    prompt_label: str | None = None,
    use_cache: bool = True,
) -> LLMJudge:
    """Create a judge for one rubric from `RUBRICS`, using the configured judge model by default.

    `prompt_label` picks the Langfuse prompt version; `use_cache` stores replies on disk.
    """
    if rubric not in RUBRICS:
        raise ValueError(f"unknown rubric {rubric}")
    if llm is None:
        from llm.llm import judge_model

        llm = judge_model()
    return LLMJudge(
        rubric,
        llm,
        model_name=get_settings().model_for("judge"),
        prompt_label=prompt_label,
        cache=disk_cache() if use_cache else None,
    )
