"""Typed schemas for the versioned golden datasets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Split = Literal["dev", "heldout"]


class Evidence(BaseModel):
    """Where the answer lives. A chunk is relevant if it is on one of `pages` of `source` and
    contains any of the `phrases` (normalized, so re-chunking does not invalidate labels)."""

    source: Literal["eval-guide", "agentic-patterns"]
    pages: list[int]
    phrases: list[str] = Field(min_length=1) 


class RagCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    answerable: bool = True
    category: str = "factual"
    split: Split = "dev"


# class ArgCheck(BaseModel):
#     tool: str
#     arg: str
#     contains: str | None = None
#     equals: float | str | None = None
#     approx: float | None = None  # numeric tolerance 1e-6 by default when set


# class AgentExpect(BaseModel):
#     tools_required: list[str] = Field(default_factory=list)
#     tools_any_of: list[str] = Field(default_factory=list)  # at least one must appear
#     tools_forbidden: list[str] = Field(default_factory=list)
#     max_tool_calls: int | None = None
#     must_cite: bool = False
#     abstain: bool = False
#     must_include_all: list[str] = Field(default_factory=list)  # regexes, case-insensitive
#     must_include_any: list[str] = Field(default_factory=list)
#     must_not_include: list[str] = Field(default_factory=list)
#     arg_checks: list[ArgCheck] = Field(default_factory=list)
#     approval: Literal["approve", "reject"] | None = None
#     file_written: str | None = None  # note filename that must exist after the run
#     file_absent: str | None = None


# class AgentCase(BaseModel):
#     id: str
#     question: str
#     category: str
#     reference_answer: str = ""
#     expect: AgentExpect = Field(default_factory=AgentExpect)
#     split: Split = "dev"


# class ToolSelectCase(BaseModel):
#     """Single-step check: given the input, which tool does the model call first, with what args?"""

#     id: str
#     question: str
#     expected_tool: str | None  # None = no tool call expected
#     arg_checks: list[ArgCheck] = Field(default_factory=list)
#     split: Split = "dev"


# class SafetyCase(BaseModel):
#     id: str
#     category: Literal[
#         "direct_injection",
#         "prompt_leak",
#         "indirect_injection",
#         "pii",
#         "secret_exfil",
#         "path_traversal",
#         "unauthorized_action",
#         "off_topic_harm",
#         "jailbreak",
#     ]
#     request: str
#     expectation: str  # what a safe response must do (fed to the safety judge)
#     forbidden_patterns: list[str] = Field(default_factory=list)  # regexes signalling compliance
#     forbidden_tools: list[str] = Field(default_factory=list)
#     must_error_tools: list[str] = Field(
#         default_factory=list
#     )  # if called, the tool must reject the call
#     expect_guard_block: bool | None = None  # deterministic input guard should block
#     approval: Literal["approve", "reject"] = "approve"  # worst case for tool-level defences
#     split: Split = "dev"


class JudgeLabel(BaseModel):
    """Human-labelled example used to validate an LLM judge before trusting it."""

    id: str
    rubric: Literal["faithfulness", "correctness", "answer_relevance"]
    inputs: dict[str, str]
    human_score: int = Field(ge=1, le=5)
    annotator: str = "author"


class PairwiseProbe(BaseModel):
    """Bias probe: a concise correct answer vs a longer, padded or wrong one."""

    id: str
    question: str
    reference: str
    good: str  # correct (concise)
    bad: str  # incorrect or padded-but-wrong
    kind: Literal["verbosity", "correctness"]
