from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from tools._format import error, ok
from utils.config import DATA_DIR

NOTES_DIR = DATA_DIR / "notes"
MAX_NOTE_CHARS = 20_000
_FILENAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\.md$")

# Names the HITL middleware must gate (side-effecting tools).
DESTRUCTIVE_TOOLS = ("save_study_notes",)


class SaveNotesInput(BaseModel):
    filename: str = Field(
        description="Lowercase file name ending in .md, letters/digits/-/_ only, e.g. 'rag-triad.md'. No paths."
    )
    markdown: str = Field(
        min_length=1, max_length=MAX_NOTE_CHARS, description="Markdown content to save."
    )


def resolve_note_path(filename: str, base: Path | None = None) -> Path:
    """Validate the file name and return a path guaranteed to be inside the notes directory."""
    if not _FILENAME.match(filename):
        raise ValueError("invalid filename")
    root = (base or NOTES_DIR).resolve()
    target = (root / filename).resolve()
    if target.parent != root:
        raise ValueError("path escapes the notes directory")
    return target


def save_study_notes(base: Path | None = None) -> BaseTool:
    def run(filename: str, markdown: str) -> str:
        try:
            path = resolve_note_path(filename, base)
        except ValueError as exc:
            return error(
                str(exc), "Use a lowercase name like 'topic.md' (a-z, 0-9, '-', '_'), no folders."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")  # idempotent: same input, same file
        return ok(saved=path.name, chars=len(markdown))

    return StructuredTool.from_function(
        func=run,
        name="save_study_notes",
        description=(
            "Save markdown study notes to the notes folder. SIDE EFFECT: writes a file and requires "
            "human approval. Use ONLY when the user explicitly asks to save notes. Do NOT use to "
            "answer questions. Returns JSON: status, saved."
        ),
        args_schema=SaveNotesInput,
    )
