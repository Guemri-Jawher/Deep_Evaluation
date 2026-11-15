"""
Subagent definitions: read from subagents.yaml, validated, then turned into deepagents `SubAgent`s.

The YAML holds only plain data. Tools and models are Python objects, so the file refers to them by
name (`tools: [search_docs]`, `model: subagent`) and `build_subagents` resolves them. An unknown
name fails at startup with a clear message instead of at the first delegation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from prompts.registry import PromptHandle

DEFAULT_PATH = Path(__file__).with_name("subagents.yaml")


class SubAgentSpec(BaseModel):
    """One entry of subagents.yaml. Unknown keys are rejected so a typo cannot pass silently."""

    model_config = ConfigDict(extra="forbid")

    description: str
    prompt: str
    tools: list[str] = Field(default_factory=list)
    model: Literal["agent", "subagent"] = "subagent"
    skills: list[str] = Field(default_factory=list)


def load_specs(path: Path = DEFAULT_PATH) -> dict[str, SubAgentSpec]:
    """Read and validate the YAML file; the key of each entry is the subagent name."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: SubAgentSpec.model_validate(cfg) for name, cfg in raw.items()}


def build_subagents(
    specs: dict[str, SubAgentSpec],
    *,
    prompts: dict[str, PromptHandle],
    tools: dict[str, BaseTool],
    models: dict[str, BaseChatModel],
) -> list[SubAgent]:
    """Resolve prompt, tool and model names to objects. Raises ValueError on an unknown tool or prompt."""
    built: list[SubAgent] = []
    for name, spec in specs.items():
        unknown = [t for t in spec.tools if t not in tools]
        if unknown:
            raise ValueError(f"subagent '{name}': unknown tools {unknown}; known: {sorted(tools)}")
        if spec.prompt not in prompts:
            raise ValueError(f"subagent '{name}': prompt '{spec.prompt}' was not loaded")
        built.append(
            {
                "name": name,
                "description": spec.description,
                "system_prompt": prompts[spec.prompt].text,
                "tools": [tools[t] for t in spec.tools],
                "model": models[spec.model],
                "skills": spec.skills,
            }
        )
    return built
