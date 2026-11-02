"""Classic RAG: retrieve once, then answer with citations. Baseline for the agentic version."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from observability.langfuse_client import get_callback_handler
from prompts.registry import PromptHandle, get_prompt
from rag.models import RetrievedChunk, extract_citations
from rag.retriever import Retriever

ABSTAIN = "The guides do not cover this."


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{r.chunk.id}] {r.chunk.text}" for r in chunks)


@dataclass
class RagResult:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    cited_ids: list[str]
    prompt: PromptHandle

    @property
    def context(self) -> str:
        return format_context(self.retrieved)

    @property
    def abstained(self) -> bool:
        return ABSTAIN.lower() in self.answer.lower()


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        *,
        k: int = 5,
        strategy: str | None = None,
        prompt_label: str | None = None,
    ) -> None:
        """
        Initialize the RAG pipeline with retriever, LLM, and optional settings.
        Args:
            retriever: The retriever to use for fetching relevant chunks.
            llm: The language model to use for generating answers.
            k: The number of top chunks to retrieve.
            strategy: The retrieval strategy to use.
            prompt_label: The label of the prompt to use from the prompt registry.
        """
        self.retriever, self.llm, self.k = retriever, llm, k
        self.strategy, self.prompt_label = strategy, prompt_label

    def answer(self, question: str) -> RagResult:
        """Answer a question using the RAG pipeline, returning the result with citations."""
        retrieved = self.retriever.retrieve(question, k=self.k, strategy=self.strategy)
        prompt = get_prompt("rag_answer", label=self.prompt_label)
        text = prompt.compile(context=format_context(retrieved), question=question)
        handler = get_callback_handler()
        config = {"callbacks": [handler]} if handler else {}
        reply = self.llm.invoke([HumanMessage(text)], config=config)  # type: ignore[arg-type]
        answer = reply.content if isinstance(reply.content, str) else str(reply.content)
        return RagResult(question, answer.strip(), retrieved, extract_citations(answer), prompt)
