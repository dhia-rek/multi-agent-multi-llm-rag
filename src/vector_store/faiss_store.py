"""Persistent FAISS vector store. Built once offline, reloaded on each app start."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.chunking_embeddings.chunker import Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)


class FaissStore:
    INDEX_FILE = "faiss.index"
    META_FILE = "meta.pkl"
    INFO_FILE = "info.json"

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings size mismatch")
        dim = embeddings.shape[1]
        # cosine similarity via inner product on normalized vectors
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        log.info("FAISS index built: %d vectors of dim %d", self.index.ntotal, dim)

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")
        faiss.write_index(self.index, str(self.index_dir / self.INDEX_FILE))
        with open(self.index_dir / self.META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        info = {
            "n_chunks": len(self.chunks),
            "dim": int(self.index.d),
            "frameworks": sorted({c.framework for c in self.chunks}),
        }
        (self.index_dir / self.INFO_FILE).write_text(json.dumps(info, indent=2))
        log.info("Index saved to %s", self.index_dir)

    def load(self) -> None:
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index not found in {self.index_dir}. "
                "Run `python -m scripts.build_index` first."
            )
        self.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        log.info("Index loaded: %d chunks", len(self.chunks))

    def search(self, query_vec: np.ndarray, k: int = 6) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not loaded.")
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec.astype("float32"), k)
        out: list[tuple[Chunk, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            out.append((self.chunks[i], float(score)))
        return out

    def info(self) -> dict:
        path = self.index_dir / self.INFO_FILE
        if path.exists():
            return json.loads(path.read_text())
        return {}
# RAG system - Apr 28
