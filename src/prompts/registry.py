"""Prompt registry: local files are the source of truth, Langfuse is the runtime registry.

* `get_prompt(name)` fetches the labelled version from Langfuse (cached client side) and falls
  back to the repository copy, so the app keeps working when Langfuse is unreachable.
* `push_defaults(label)` publishes the repository copies as new Langfuse versions (idempotent:
  unchanged prompts do not create a new version). Promote a version by moving its label in the UI.
* Every handle exposes `name` and `version`, which are attached to traces for A/B comparison.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from observability.langfuse_client import get_langfuse
from utils.config import PROMPT_DEFAULTS_DIR, get_settings

log = logging.getLogger(__name__)
_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class PromptHandle:
    name: str
    text: str
    version: int | None
    label: str | None
    source: str  # "langfuse" | "local"
    client: Any = None  # underlying Langfuse prompt object, used for trace linking

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.text))

    def compile(self, **values: object) -> str:
        missing = self.variables - values.keys()
        if missing:
            raise KeyError(f"prompt '{self.name}' missing variables: {sorted(missing)}")
        return _VAR.sub(lambda m: str(values[m.group(1)]), self.text)

    @property
    def tag(self) -> str:
        return f"{self.name}@{self.version if self.version is not None else 'local'}"


PROMPT_DIRS = (
    PROMPT_DEFAULTS_DIR,
    PROMPT_DEFAULTS_DIR.parent / "judge",
)  # app prompts, judge rubrics


def local_prompt_names() -> list[str]:
    return sorted(p.stem for d in PROMPT_DIRS for p in d.glob("*.md"))


def read_local(name: str) -> str:
    for directory in PROMPT_DIRS:
        path = directory / f"{name}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"no local prompt named '{name}' in {[str(d) for d in PROMPT_DIRS]}")


def get_prompt(name: str, *, label: str | None = None) -> PromptHandle:
    local = read_local(name)
    lf = get_langfuse()
    if lf is None:
        return PromptHandle(name, local, None, None, "local")
    label = label or get_settings().langfuse_prompt_label
    try:
        client = lf.get_prompt(name, label=label, type="text", fallback=local)
    except Exception:
        log.warning("Langfuse prompt fetch failed for %s; using local copy", name, exc_info=True)
        return PromptHandle(name, local, None, None, "local")
    if getattr(client, "is_fallback", False):
        return PromptHandle(name, local, None, None, "local")
    # Langfuse templates use the same {{var}} syntax, so the text can be compiled locally too.
    return PromptHandle(name, client.prompt, client.version, label, "langfuse", client)


def push_defaults(label: str = "staging", *, names: list[str] | None = None) -> dict[str, str]:
    """Create new Langfuse prompt versions from the repository copies. Returns name -> action."""
    lf = get_langfuse()
    if lf is None:
        raise RuntimeError("Langfuse is not configured (set LANGFUSE_PUBLIC_KEY/SECRET_KEY).")
    result: dict[str, str] = {}
    for name in names or local_prompt_names():
        text = read_local(name)
        try:
            current = lf.get_prompt(name, label="latest", type="text", max_retries=0)
            if current.prompt.strip() == text:
                result[name] = f"unchanged (v{current.version})"
                continue
        except Exception:
            pass  # prompt does not exist yet
        created = lf.create_prompt(
            name=name,
            type="text",
            prompt=text,
            labels=[label],
            commit_message="sync from repository",
        )
        result[name] = f"created v{created.version} [{label}]"
    return result
