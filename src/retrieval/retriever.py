"""High-level RAG retriever. Wraps embedder + FAISS store."""
from __future__ import annotations

from dataclasses import dataclass

from src.chunking_embeddings.chunker import Chunk
from src.chunking_embeddings.embedder import Embedder
from src.utils.config import settings
from src.utils.logging import get_logger
from src.vector_store.faiss_store import FaissStore

log = get_logger(__name__)
FILTER_OVERFETCH_FACTOR = 4  # leave room for filtered-out frameworks


@dataclass
class RetrievedPassage:
    """A chunk returned by a search together with its similarity score."""

    chunk: Chunk
    score: float

    def to_context_block(self) -> str:
        """Render as ``[framework | title | p.N]`` header followed by the chunk text."""
        return (
            f"[{self.chunk.framework} | {self.chunk.title} | p.{self.chunk.page}]\n"
            f"{self.chunk.text}"
        )


class Retriever:
    """Embeds queries and searches the FAISS store for matching chunks."""

    def __init__(self, store: FaissStore | None = None, embedder: Embedder | None = None) -> None:
        """Use the given store/embedder or create defaults; loads the index from disk if the store is empty.

        Raises:
            FileNotFoundError: If the store is empty and no saved index exists.
        """
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
        """Return the top passages for one query.

        Args:
            query: Natural-language query.
            k: Number of passages; defaults to ``settings.top_k``.
            framework_filter: If given, keep only chunks whose framework is listed.
                Over-fetches before filtering, so fewer than ``k`` may still be returned.

        Returns:
            Up to ``k`` passages, best score first.
        """
        k = k or settings.top_k
        # Over-fetch then filter, so framework_filter still returns k items.
        fetch_k = k * FILTER_OVERFETCH_FACTOR if framework_filter else k
        qvec = self.embedder.encode([query])
        hits = self.store.search(qvec, k=fetch_k)
        passages = [RetrievedPassage(chunk=c, score=s) for c, s in hits]
        if framework_filter:
            allowed = set(framework_filter)
            passages = [p for p in passages if p.chunk.framework in allowed]
        return passages[:k]

    def retrieve_many(self, queries: list[str], k: int | None = None) -> list[list[RetrievedPassage]]:
        """Retrieve top-k passages for each query, embedding all queries in one batch.

        Args:
            queries: Query strings.
            k: Passages per query; defaults to ``settings.top_k``.

        Returns:
            One passage list per query, in input order (empty list if no queries).
        """
        k = k or settings.top_k
        if not queries:
            return []
        qvecs = self.embedder.encode(queries)
        return [
            [RetrievedPassage(chunk=c, score=sc) for c, sc in self.store.search(qvecs[i], k=k)]
            for i in range(len(queries))
        ]

    @staticmethod
    def format_context(passages: list[RetrievedPassage]) -> str:
        """Join passages' context blocks with ``---`` separators for use in a prompt."""
        return "\n\n---\n\n".join(p.to_context_block() for p in passages)
