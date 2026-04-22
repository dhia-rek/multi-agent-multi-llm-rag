"""Local sentence-transformer embedder. No API call, runs on CPU."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class Embedder:
    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model

    @property
    def model(self) -> SentenceTransformer:
        if Embedder._model is None:
            log.info("Loading embedding model: %s", self.model_name)
            Embedder._model = SentenceTransformer(self.model_name)
        return Embedder._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
