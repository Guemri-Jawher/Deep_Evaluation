"""Offline tests of how the agent is wired: config contract, subagents, budgets and safety settings.

`create_deep_agent` is replaced by a recorder, so no model or network is used. These tests check
what we pass to it; they do not check what the LLM does (see scripts/try_agent.py for that).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agent.agent as agent_module
from agent.subagent_config import build_subagents, load_specs
from rag.models import Chunk, RetrievedChunk


@pytest.fixture
def captured(monkeypatch):
    """Build the agent with fake models and return the arguments given to create_deep_agent."""
    calls: dict = {}
    monkeypatch.setattr("prompts.registry.get_langfuse", lambda: None)  # local prompts only
    monkeypatch.setattr(
        agent_module, "create_deep_agent", lambda **kwargs: calls.update(kwargs) or "GRAPH"
    )
    retriever = SimpleNamespace(
        retrieve=lambda *a, **k: [RetrievedChunk(Chunk("g:p1:c0", "t", "g", 1, 0), 1.0, 1)]
    )
    built = agent_module.build_agent(
        model=FakeListChatModel(responses=["x"]),
        subagent_model=FakeListChatModel(responses=["y"]),
        retriever=retriever,
    )
    calls["built"] = built
    return calls


def test_agent_has_the_expected_tools(captured):
    assert captured["built"].tool_names == [
        "calculator",
        "pass_k",
        "save_study_notes",
        "search_docs",
    ]


def test_agent_has_two_named_subagents(captured):
    names = {s["name"] for s in captured["subagents"]}
    assert names == {"researcher", "grounding-critic"}


def test_subagents_are_read_only_and_use_the_subagent_model(captured):
    for sub in captured["subagents"]:
        assert [t.name for t in sub["tools"]] == ["search_docs"]
        assert sub["model"].responses == ["y"]
        assert sub["system_prompt"]


def test_prompt_versions_are_tracked_for_every_prompt(captured):
    assert set(captured["built"].prompt_versions) == {
        "agent_system",
        "researcher_system",
        "critic_system",
    }


def test_notes_require_human_approval(captured):
    assert captured["interrupt_on"] == {"save_study_notes": True}


def test_skills_folder_cannot_be_written(captured):
    deny = [p for p in captured["permissions"] if p.mode == "deny"]
    assert any("/skills/**" in p.paths and "write" in p.operations for p in deny)


def test_budgets_are_enforced_by_middleware(captured):
    kinds = {type(m).__name__ for m in captured["middleware"]}
    assert {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"} <= kinds
    assert captured["built"].recursion_limit > 0


def test_a_checkpointer_is_always_present(captured):
    assert captured["checkpointer"] is not None


# --- subagents.yaml ---------------------------------------------------------------------------


def test_yaml_specs_load():
    specs = load_specs()
    assert set(specs) == {"researcher", "grounding-critic"}
    assert all(s.tools == ["search_docs"] for s in specs.values())


def test_unknown_tool_in_yaml_fails_at_startup():
    with pytest.raises(ValueError, match="unknown tools"):
        build_subagents(
            load_specs(),
            prompts={
                "researcher_system": SimpleNamespace(text="p"),
                "critic_system": SimpleNamespace(text="p"),
            },
            tools={},
            models={"agent": object(), "subagent": object()},
        )
