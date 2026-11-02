"""Typed runtime configuration. All knobs come from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SKILLS_DIR = PACKAGE_ROOT / "skills"
PROMPT_DEFAULTS_DIR = PACKAGE_ROOT / "prompts" / "defaults"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider: LLM_PROVIDER picks which block below is used for every chat model
    llm_provider: Literal["openrouter", "nvidia"] = "openrouter"

    # OpenRouter
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    agent_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    subagent_model: str = "google/gemma-4-26b-a4b-it:free"
    judge_model: str = "google/gemma-4-31b-it:free"
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 4
    # Comma-separated; OpenRouter tries these in order when the primary model errors (e.g. 429).
    fallback_models: str = "poolside/laguna-s-2.1:free,nvidia/nemotron-3-ultra-550b-a55b:free"

    # NVIDIA API catalog (OpenAI-compatible, key starts with "nvapi-")
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_agent_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_subagent_model: str = "openai/gpt-oss-20b"
    nvidia_judge_model: str = "openai/gpt-oss-20b"

    # Embeddings / vector store
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: Path = DATA_DIR / "chroma"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_strategy: str = Field(default="hybrid", pattern="^(vector|bm25|hybrid)$")

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_prompt_label: str = "production"

    # Runtime budgets (enforced in code)
    max_steps: int = 25
    max_tool_calls: int = 20
    max_tool_output_chars: int = 6000
    repeat_tool_call_limit: int = 2

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key.get_secret_value())

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.get_secret_value())

    @property
    def llm_api_key(self) -> SecretStr:
        return self.nvidia_api_key if self.llm_provider == "nvidia" else self.openrouter_api_key

    @property
    def llm_api_key_env(self) -> str:
        return "NVIDIA_API_KEY" if self.llm_provider == "nvidia" else "OPENROUTER_API_KEY"

    @property
    def llm_base_url(self) -> str:
        return self.nvidia_base_url if self.llm_provider == "nvidia" else self.openrouter_base_url

    def model_for(self, role: Literal["agent", "subagent", "judge"]) -> str:
        """Model name for a role on the selected provider (names differ between providers)."""
        if self.llm_provider == "nvidia":
            names = {
                "agent": self.nvidia_agent_model,
                "subagent": self.nvidia_subagent_model,
                "judge": self.nvidia_judge_model,
            }
        else:
            names = {
                "agent": self.agent_model,
                "subagent": self.subagent_model,
                "judge": self.judge_model,
            }
        return names[role]


@lru_cache
def get_settings() -> Settings:
    return Settings()
