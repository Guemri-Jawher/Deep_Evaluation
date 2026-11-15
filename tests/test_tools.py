"""Offline tests of the agent tools: typed inputs, safe evaluation, path safety and error contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag.models import Chunk, RetrievedChunk
from tools.docs import make_search_docs
from tools.math_tools import calculator, pass_k, safe_eval
from tools.notes import DESTRUCTIVE_TOOLS, resolve_note_path, save_study_notes


def call(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


# --- calculator -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", 14.0), ("0.8**3", 0.512), ("round(1-(1-0.7)**5, 4)", 0.9976), ("-(2+3)", -5.0)],
)
def test_safe_eval_computes(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x')", "(1).__class__", "x + 1", "2**100000"],
)
def test_safe_eval_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calculator_returns_structured_errors():
    assert call(calculator(), expression="1/0")["status"] == "error"
    assert call(calculator(), expression="import os")["status"] == "error"
    assert call(calculator(), expression="2+2")["result"] == 4.0


def test_pass_k_formulas():
    out = call(pass_k(), success_rate=0.8, k=3)
    assert out["pass_pow_k"] == pytest.approx(0.512)
    assert out["pass_at_k"] == pytest.approx(0.992)


def test_pass_k_validates_range():
    with pytest.raises(ValidationError):
        pass_k().invoke({"success_rate": 1.5, "k": 3})
    with pytest.raises(ValidationError):
        pass_k().invoke({"success_rate": 0.5, "k": 0})


# --- notes (the only side-effecting tool) -----------------------------------------------------


def test_notes_tool_is_gated():
    assert "save_study_notes" in DESTRUCTIVE_TOOLS


@pytest.mark.parametrize(
    "name", ["../escape.md", "a/b.md", "UPPER.md", ".env", "x.txt", "", "a" * 70 + ".md"]
)
def test_note_filename_rejects_paths_and_odd_names(name, tmp_path):
    with pytest.raises(ValueError):
        resolve_note_path(name, tmp_path)


def test_note_saved_inside_the_notes_folder(tmp_path):
    out = call(save_study_notes(tmp_path), filename="rag-triad.md", markdown="# Notes")
    assert out["status"] == "ok"
    assert (tmp_path / "rag-triad.md").read_text(encoding="utf-8") == "# Notes"


def test_note_with_bad_name_writes_nothing(tmp_path):
    out = call(save_study_notes(tmp_path), filename="../x.md", markdown="# Notes")
    assert out["status"] == "error"
    assert list(tmp_path.iterdir()) == []


def test_note_content_is_size_limited(tmp_path):
    with pytest.raises(ValidationError):
        save_study_notes(tmp_path).invoke({"filename": "big.md", "markdown": "x" * 20_001})


# --- search_docs ------------------------------------------------------------------------------


def stub_retriever(hits):
    return SimpleNamespace(retrieve=lambda query, k=5, source=None, **_: hits)


def hit(chunk_id="eval-guide:p6:c1", text="1. baseline ... 4. decide"):
    source, page, _ = chunk_id.split(":")
    chunk = Chunk(chunk_id, text, source, int(page[1:]), 1)
    return RetrievedChunk(chunk, 0.9, 1, "hybrid")


def test_search_docs_returns_ids_and_text():
    out = call(make_search_docs(stub_retriever([hit()])), query="eval driven loop")
    assert out["status"] == "ok"
    assert out["results"][0]["id"] == "eval-guide:p6:c1"
    assert out["results"][0]["page"] == 6


def test_search_docs_no_hits_explains_how_to_recover():
    out = call(make_search_docs(stub_retriever([])), query="nothing matches this")
    assert out["status"] == "error"
    assert "hint" in out


def test_search_docs_output_is_bounded():
    huge = hit(text="x" * 5000)
    raw = make_search_docs(stub_retriever([huge] * 8)).invoke({"query": "long output", "k": 8})
    assert len(raw) <= 6000  # MAX_TOOL_OUTPUT_CHARS default


@pytest.mark.parametrize(
    "args",
    [
        {"query": "ab"},
        {"query": "valid query", "k": 20},
        {"query": "valid query", "source": "other-guide"},
    ],
)
def test_search_docs_validates_input(args):
    with pytest.raises(ValidationError):
        make_search_docs(stub_retriever([])).invoke(args)
