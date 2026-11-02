"""Index (Chroma + BM25 over the same chunks) and the retrieval strategies compared in the evals."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.bm25 import BM25Client, reciprocal_rank_fusion
from rag.embeddings import get_embeddings
from rag.ingest import load_corpus
from rag.models import Chunk, RetrievedChunk
from utils.config import Settings, get_settings

STRATEGIES = ("vector", "bm25", "hybrid")


def _fingerprint(chunks: Sequence[Chunk], settings: Settings) -> str:
    """Compute a fingerprint for the given chunks and settings."""
    h = hashlib.sha256(settings.embedding_model.encode())
    for c in chunks:
        h.update(c.id.encode())
        h.update(hashlib.sha256(c.text.encode()).digest())
    return h.hexdigest()


def _collection_name(settings: Settings) -> str:
    """Generate a collection name for the given settings."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", settings.embedding_model).strip("_")
    return f"kb_{slug}"[:60]


class Retriever:
    """Retrieves chunks with `vector`, `bm25` or `hybrid` (reciprocal rank fusion)."""

    def __init__(
        self, chunks: Sequence[Chunk], store: Chroma | None, default_strategy: str
    ) -> None:
        self.chunks = list(chunks)
        self._by_id = {c.id: c for c in self.chunks}
        self._store = store
        self._bm25 = BM25Client(
            [
                Document(page_content=c.text, metadata={"chunk_id": c.id, "source": c.source})
                for c in self.chunks
            ]
        )
        self.default_strategy = default_strategy

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        source: str | None = None,
        strategy: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks for a query using the specified strategy."""
        strategy = strategy or self.default_strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")
        pool = max(k * 4, 20)
        vec = self._vector(query, pool, source) if strategy in ("vector", "hybrid") else []
        lex = self._lexical(query, pool, source) if strategy in ("bm25", "hybrid") else []
        if strategy == "vector":
            ranked = vec
        elif strategy == "bm25":
            ranked = lex
        else:
            fused = reciprocal_rank_fusion([[i for i, _ in vec], [i for i, _ in lex]])
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievedChunk(self._by_id[cid], float(score), rank, strategy)
            for rank, (cid, score) in enumerate(ranked[:k], start=1)
        ]

    def _vector(self, query: str, k: int, source: str | None) -> list[tuple[str, float]]:
        """Retrieve the top-k chunks for a query using the vector index."""
        if self._store is None:
            raise RuntimeError("vector index is not available for this retriever")
        flt = {"source": source} if source else None
        hits = self._store.similarity_search_with_score(query, k=k, filter=flt)
        return [(doc.metadata["chunk_id"], 1.0 - float(dist)) for doc, dist in hits]

    def _lexical(self, query: str, k: int, source: str | None) -> list[tuple[str, float]]:
        """Retrieve the top-k chunks for a query using the lexical (BM25) index."""
        hits = self._bm25.search(query, k=k, source=source)
        return [(doc.metadata["chunk_id"], score) for doc, score in hits]


def build_index(*, rebuild: bool = False, settings: Settings | None = None) -> Retriever:
    """Chunk the corpus and (re)build the persistent vector index when the corpus changed."""
    s = settings or get_settings()
    chunks = load_corpus()
    persist: Path = s.chroma_dir
    persist.mkdir(parents=True, exist_ok=True)
    manifest_path = persist / "manifest.json"
    fingerprint = _fingerprint(chunks, s)

    from chromadb.config import Settings as ChromaSettings

    store = Chroma(
        collection_name=_collection_name(s),
        embedding_function=get_embeddings(),
        persist_directory=str(persist),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=ChromaSettings(
            anonymized_telemetry=False, is_persistent=True, persist_directory=str(persist)
        ),
    )
    current = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if rebuild or current.get("fingerprint") != fingerprint:
        store.reset_collection()
        batch = 64
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            store.add_documents(
                [
                    Document(
                        page_content=c.text,
                        metadata={"chunk_id": c.id, "source": c.source, "page": c.page},
                    )
                    for c in part
                ],
                ids=[c.id for c in part],
            )
        manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "chunks": len(chunks)}))
    return Retriever(chunks, store, s.retrieval_strategy)


@lru_cache
def get_retriever() -> Retriever:
    return build_index()
