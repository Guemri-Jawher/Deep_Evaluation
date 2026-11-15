"""Builds the Deep Agent: supervisor + two subagents, scoped skills, guardrails and budgets.

Design notes (see README for the full rationale):
* Pattern: supervisor-worker with agentic RAG. Planning/todo middleware is left out on purpose:
  questions here are short, so extra autonomy would add cost without evidence it helps (the evals decide).
* Least privilege: the virtual filesystem is in-memory scratch space. Skills are mounted read-only
  from the package; nothing else on disk is reachable by the agent.
* Budgets live in middleware (steps, tool calls, repeated calls), not in the prompt.
* The only side-effecting tool, `save_study_notes`, is gated by human-in-the-loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agent.subagent_config import build_subagents, load_specs

from guardrails.middleware import ToolPolicyMiddleware
from prompts.registry import PromptHandle, get_prompt
from rag.retriever import Retriever, get_retriever
from tools.notes import DESTRUCTIVE_TOOLS
from tools.registry import build_tools
from utils.config import SKILLS_DIR, get_settings

MAIN_SKILLS = "/skills/main/"


@dataclass
class BuiltAgent:
    graph: Any
    system_prompt: PromptHandle
    prompts: dict[str, PromptHandle]
    tool_names: list[str]
    recursion_limit: int
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_versions(self) -> dict[str, str]:
        return {name: p.tag for name, p in self.prompts.items()}


def build_backend() -> CompositeBackend:
    """In-memory scratch space plus a read-only mount of the package skills directory."""
    return CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )


def build_agent(
    *,
    model: BaseChatModel | None = None,
    subagent_model: BaseChatModel | None = None,
    retriever: Retriever | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    prompt_label: str | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> BuiltAgent:
    s = get_settings()
    if model is None or subagent_model is None:
        from llm.llm import agent_model
        from llm.llm import subagent_model as make_sub

        model = model or agent_model()
        subagent_model = subagent_model or make_sub()

    retriever = retriever or get_retriever()
    tools = build_tools(retriever) + (extra_tools or [])
    by_name = {t.name: t for t in tools}

    specs = load_specs()
    prompts = {"agent_system": get_prompt("agent_system", label=prompt_label)}
    for spec in specs.values():
        prompts[spec.prompt] = get_prompt(spec.prompt, label=prompt_label)

    subagents = build_subagents(
        specs,
        prompts=prompts,
        tools=by_name,
        models={"agent": model, "subagent": subagent_model},
    )

    middleware: list[Any] = [
        ModelCallLimitMiddleware(run_limit=s.max_steps, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=s.max_tool_calls, exit_behavior="continue"),
        ToolPolicyMiddleware(
            allowed_tools=by_name.keys(),
            repeat_limit=s.repeat_tool_call_limit,
            max_output_chars=s.max_tool_output_chars,
        ),
    ]
    graph = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=prompts["agent_system"].text,
        subagents=subagents,
        skills=[MAIN_SKILLS],
        backend=build_backend(),
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        middleware=middleware,
        interrupt_on={name: True for name in DESTRUCTIVE_TOOLS},
        checkpointer=checkpointer or InMemorySaver(),
        name="kb-supervisor",
    )
    return BuiltAgent(
        graph=graph,
        system_prompt=prompts["agent_system"],
        prompts=prompts,
        tool_names=sorted(by_name),
        recursion_limit=s.max_steps * 4,
    )
