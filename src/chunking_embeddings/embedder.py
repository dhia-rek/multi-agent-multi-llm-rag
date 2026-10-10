"""Local sentence-transformer embedder. No API call, runs on CPU."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class Embedder:
    """Lazy wrapper around a SentenceTransformer producing normalized float32 vectors."""

    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str | None = None) -> None:
        """Set the model name (default ``settings.embedding_model``); the model loads on first use."""
        self.model_name = model_name or settings.embedding_model

    @property
    def model(self) -> SentenceTransformer:
        """The SentenceTransformer, loaded lazily and cached on the class.

        Shared by all instances regardless of ``model_name``.
        """
        if Embedder._model is None:
            log.info("Loading embedding model: %s", self.model_name)
            Embedder._model = SentenceTransformer(self.model_name)
        return Embedder._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into L2-normalized float32 vectors (so inner product = cosine).

        Args:
            texts: Strings to embed.

        Returns:
            Array of shape (len(texts), dim).
        """
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    @property
    def dim(self) -> int:
        """Embedding dimension of the loaded model."""
        dim = self.model.get_sentence_embedding_dimension()
        assert dim is not None  # always set for a loaded SentenceTransformer
        return dim
