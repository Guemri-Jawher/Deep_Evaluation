"""Single place that builds chat models, so providers can be swapped in one line.

LLM_PROVIDER selects the provider (`openrouter` or `nvidia`); both speak the OpenAI protocol.
OpenRouter's free `:free` models are rate-limited, hence the retries and timeouts configured here.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.config import get_settings


def build_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning: bool = True,
) -> BaseChatModel:
    s = get_settings()
    if not s.llm_enabled:
        raise RuntimeError(
            f"{s.llm_api_key_env} is not set (LLM_PROVIDER={s.llm_provider}). "
            "Copy .env.example to .env and fill it."
        )
    # The `reasoning` switch is an OpenRouter parameter; other providers get a plain request.
    no_reasoning = not reasoning and s.llm_provider == "openrouter"
    return ChatOpenAI(
        model=model or s.model_for("agent"),
        api_key=s.llm_api_key,
        base_url=s.llm_base_url,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        extra_body={"reasoning": {"effort": "none"}} if no_reasoning else None,
    )


def agent_model() -> BaseChatModel:
    return build_chat_model(get_settings().model_for("agent"))


def subagent_model() -> BaseChatModel:
    """Build the subagent chat model. Used for handling sub-tasks within the agent framework."""
    return build_chat_model(get_settings().model_for("subagent"))


def judge_model() -> BaseChatModel:
    """Judges are deterministic and use a different model family than the agent.

    Reasoning is off: a reasoning model spends the token cap on thinking and truncates the JSON.
    """
    return build_chat_model(
        get_settings().model_for("judge"), temperature=0.0, max_tokens=2000, reasoning=False
    )
