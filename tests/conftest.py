"""Shared offline fixtures: no network, no API keys, no embedding model download."""
from __future__ import annotations

import dataclasses
import hashlib
import re
import sys
import types

import numpy as np
import pytest

# src.chunking_embeddings.embedder imports sentence_transformers at module level
# (pulls in torch). Tests never use the real Embedder, so stub the module.
if "sentence_transformers" not in sys.modules:
    _stub = types.ModuleType("sentence_transformers")
    _stub.SentenceTransformer = object  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = _stub

from src.chunking_embeddings.chunker import Chunk  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.routing.router import Router  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.vector_store.faiss_store import FaissStore  # noqa: E402

DIM = 32


class FakeEmbedder:
    """Deterministic hashed bag-of-words -> L2-normalised float32 vectors."""

    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


def make_settings(**overrides):
    """Copy of the frozen global settings with fields replaced."""
    return dataclasses.replace(settings, **overrides)


@pytest.fixture
def demo_settings(monkeypatch):
    """Force demo mode in the router module; return the patched settings."""
    s = make_settings(demo_mode="on")
    monkeypatch.setattr("src.routing.router.settings", s)
    return s


@pytest.fixture
def tiny_chunks() -> list[Chunk]:
    texts = [
        ("dt_framework", "Why what how: start with the purpose of digital transformation."),
        ("dt_framework", "Customer centricity and online booking improve the repair experience."),
        ("bikeshop_roadmap", "Shopify store with a point of sale and two physical shops."),
        ("bikeshop_roadmap", "Staff training and digital leadership culture workshops."),
    ]
    return [
        Chunk(f"c{i:05d}", text, "/fake/doc.pdf", fw, f"doc{i}", i + 1)
        for i, (fw, text) in enumerate(texts)
    ]


@pytest.fixture
def fake_retriever(tmp_path, tiny_chunks) -> Retriever:
    embedder = FakeEmbedder()
    store = FaissStore(tmp_path / "index")
    store.build(tiny_chunks, embedder.encode([c.text for c in tiny_chunks]))
    return Retriever(store=store, embedder=embedder)


@pytest.fixture
def orchestrator(tmp_path, monkeypatch, fake_retriever):
    """Demo-mode Orchestrator over the tiny index, cache isolated in tmp_path."""
    from src.cache.disk_cache import DiskCache
    from src.orchestrator.pipeline import Orchestrator

    s = make_settings(demo_mode="on", cache_dir=tmp_path / "cache")
    monkeypatch.setattr("src.routing.router.settings", s)
    monkeypatch.setattr("src.cache.disk_cache.settings", s)
    monkeypatch.setattr("src.orchestrator.graph.settings", s)
    orch = Orchestrator(router=Router(), retriever=fake_retriever)
    assert isinstance(orch.cache, DiskCache)
    return orch
