"""Offline index build. Run this once after dropping PDFs into corpus/.
Embeddings + FAISS index are persisted to vector_store/index/.

Usage:
    python -m scripts.build_index
"""
from __future__ import annotations

from src.chunking_embeddings.chunker import chunk_documents
from src.chunking_embeddings.embedder import Embedder
from src.data_ingestion.loader import load_corpus
from src.utils.config import settings
from src.utils.logging import get_logger
from src.vector_store.faiss_store import FaissStore

log = get_logger(__name__)


def main() -> None:
    log.info("Loading corpus from %s", settings.corpus_dir)
    docs = load_corpus(settings.corpus_dir)

    log.info("Chunking (size=%d, overlap=%d)", settings.chunk_size, settings.chunk_overlap)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    log.info("Produced %d chunks", len(chunks))

    embedder = Embedder()
    log.info("Embedding %d chunks with %s", len(chunks), embedder.model_name)
    vectors = embedder.encode([c.text for c in chunks])

    store = FaissStore(settings.index_dir)
    store.build(chunks, vectors)
    store.save()
    log.info("Done. Index → %s", settings.index_dir)


if __name__ == "__main__":
    main()
# Build scripts - May 8
