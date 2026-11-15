from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from rag.retriever import Retriever
from tools._format import error, ok, truncate
from utils.config import get_settings

MAX_PASSAGE_CHARS = 700


class SearchDocsInput(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=300,
        description="Short keyword-rich search query, e.g. 'recall at k retrieval metric'.",
    )
    source: Literal["eval-guide", "agentic-patterns"] | None = Field(
        default=None,
        description="Restrict to one guide: 'eval-guide' (evaluation) or 'agentic-patterns' (agents).",
    )
    k: int = Field(default=5, ge=1, le=8, description="Number of passages to return (1-8).")


def make_search_docs(retriever: Retriever) -> BaseTool:
    def search_docs(query: str, source: str | None = None, k: int = 5) -> str:
        hits = retriever.retrieve(query, k=k, source=source)
        if not hits:
            return error(
                "no passages matched",
                "Rewrite the query with the guide's own terminology, or drop the source filter.",
            )
        limit = get_settings().max_tool_output_chars
        results = [
            {
                "id": h.chunk.id,
                "source": h.chunk.source,
                "page": h.chunk.page,
                "text": truncate(h.chunk.text, MAX_PASSAGE_CHARS),
            }
            for h in hits
        ]
        payload = ok(
            query=query,
            count=len(results),
            results=results,
            next_step="Cite passages as [id]. Retrieve again with a rewritten query if weak.",
        )
        return truncate(payload, limit)

    return StructuredTool.from_function(
        func=search_docs,
        name="search_docs",
        description=(
            "Search the two guides (evaluation field guide, agentic design patterns guide) and return "
            "the most relevant passages with ids. Use for ANY question about evaluation, RAG, LLM "
            "judges, agents, tools, patterns or deployment. Do NOT use for arithmetic. Read-only. "
            "Returns JSON: status, results[{id, source, page, text}]. Cite ids as [id]."
        ),
        args_schema=SearchDocsInput,
    )
