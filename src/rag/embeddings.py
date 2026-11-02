"""Local embeddings (no API key) behind the LangChain `Embeddings` interface."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from utils.config import DATA_DIR, get_settings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        cache = DATA_DIR / "cache" / "fastembed"
        cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))
        self._is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query into a vector."""
        query = f"{_BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        return [float(x) for x in next(iter(self._model.embed([query])))]


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(get_settings().embedding_model)
