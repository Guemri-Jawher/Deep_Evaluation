"""Load versioned JSONL golden sets and verify they were not changed without a version bump."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from datasets.utils.schema import (
    # AgentCase,
    JudgeLabel,
    PairwiseProbe,
    RagCase,
    # SafetyCase,
    # ToolSelectCase,
)
from pydantic import BaseModel

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_VERSION = "v1"
M = TypeVar("M", bound=BaseModel)

FILES: dict[str, type[BaseModel]] = {
    "rag_golden": RagCase,
    # "agent_tasks": AgentCase,
    # "tool_select": ToolSelectCase,
    # "safety_cases": SafetyCase,
    "judge_labels": JudgeLabel,
    "pairwise_probes": PairwiseProbe,
}


def dataset_dir(version: str = DEFAULT_VERSION) -> Path:
    return DATA_ROOT / version


def dataset_path(name: str, version: str = DEFAULT_VERSION) -> Path:
    """Find `<name>.jsonl` under the version folder, including subfolders such as `rag/`."""
    matches = sorted(dataset_dir(version).rglob(f"{name}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"{name}.jsonl not found under {dataset_dir(version)}")
    return matches[0]


def load_jsonl(name: str, model: type[M], version: str = DEFAULT_VERSION) -> list[M]:
    path = dataset_path(name, version)
    rows: list[M] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path.name}:{lineno}: {exc}") from exc
    ids = [getattr(r, "id") for r in rows]  # noqa: B009
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate case ids")
    return rows


def load_cases(name: str, *, split: str | None = None, version: str = DEFAULT_VERSION) -> list:  # type: ignore[type-arg]
    """Load a dataset by name; `split` may be 'dev', 'heldout' or None for both."""
    rows = load_jsonl(name, FILES[name], version)
    if split is None:
        return rows
    return [r for r in rows if getattr(r, "split", "dev") == split]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_manifest(version: str = DEFAULT_VERSION) -> dict[str, Any]:
    directory = dataset_dir(version)
    return {
        "version": version,
        "files": {p.name: file_hash(p) for p in sorted(directory.rglob("*.jsonl"))},
    }


def write_manifest(version: str = DEFAULT_VERSION) -> Path:
    path = dataset_dir(version) / "manifest.yaml"
    path.write_text(yaml.safe_dump(compute_manifest(version), sort_keys=True), encoding="utf-8")
    return path


def verify_manifest(version: str = DEFAULT_VERSION) -> list[str]:
    """Return a list of drift problems (empty = datasets match the frozen manifest)."""
    path = dataset_dir(version) / "manifest.yaml"
    if not path.exists():
        return ["manifest.yaml missing: run `kb-eval datasets freeze`"]
    frozen: dict[str, str] = yaml.safe_load(path.read_text(encoding="utf-8"))["files"]
    current: dict[str, str] = compute_manifest(version)["files"]
    problems = [
        f"{n}: changed without re-freezing" for n, h in frozen.items() if current.get(n) != h
    ]
    problems += [f"{n}: not in manifest" for n in current if n not in frozen]
    return problems
