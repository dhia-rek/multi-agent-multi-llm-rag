"""High-level RAG retriever. Wraps embedder + FAISS store."""
from __future__ import annotations

from dataclasses import dataclass

from src.chunking_embeddings.chunker import Chunk
from src.chunking_embeddings.embedder import Embedder
from src.utils.config import settings
from src.utils.logging import get_logger
from src.vector_store.faiss_store import FaissStore

log = get_logger(__name__)


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float

    def to_context_block(self) -> str:
        return (
            f"[{self.chunk.framework} | {self.chunk.title} | p.{self.chunk.page}]\n"
            f"{self.chunk.text}"
        )


class Retriever:
    def __init__(self, store: FaissStore | None = None, embedder: Embedder | None = None):
        self.store = store or FaissStore(settings.index_dir)
        if not self.store.chunks:
            self.store.load()
        self.embedder = embedder or Embedder()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        framework_filter: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        k = k or settings.top_k
        # Over-fetch then filter, so framework_filter still returns k items.
        fetch_k = k * 4 if framework_filter else k
        qvec = self.embedder.encode([query])
        hits = self.store.search(qvec, k=fetch_k)
        passages = [RetrievedPassage(chunk=c, score=s) for c, s in hits]
        if framework_filter:
            allowed = set(framework_filter)
            passages = [p for p in passages if p.chunk.framework in allowed]
        return passages[:k]

    @staticmethod
    def format_context(passages: list[RetrievedPassage]) -> str:
        return "\n\n---\n\n".join(p.to_context_block() for p in passages)
