"""Adapter that lets DeepEval metrics use any LangChain chat model (OpenRouter judges) as the grader."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from observability.langfuse_client import get_callback_handler


def extract_json(text: str) -> Any:
    """First top-level JSON object in `text` (free models often wrap JSON in prose or fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class LangChainJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    def __init__(self, llm: BaseChatModel, name: str = "judge") -> None:
        self._llm = llm
        super().__init__(model=name)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._llm

    def get_model_name(self) -> str:
        return str(self.name)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        if schema is not None:
            prompt += (
                "\n\nRespond with ONLY a JSON object that matches this JSON schema, no prose:\n"
                + json.dumps(schema.model_json_schema())
            )
        handler = get_callback_handler()  # None without Langfuse keys
        config: Any = {"run_name": "deepeval-judge", "tags": ["judge"]}
        if handler:
            config["callbacks"] = [handler]
        reply = self._llm.invoke(prompt, config=config)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        if not text.strip():
            md = reply.response_metadata
            raise ValueError(
                f"judge returned empty text (finish_reason={md.get('finish_reason')}, "
                f"completion_tokens={(md.get('token_usage') or {}).get('completion_tokens')}); "
                "a reasoning model may have used its whole token cap on thinking"
            )
        return schema.model_validate(extract_json(text)) if schema is not None else text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)
