"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    index_dir: Path = PROJECT_ROOT / "vector_store" / "index"
    cache_dir: Path = PROJECT_ROOT / ".cache"

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    gemini_fast_model: str = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")
    gemini_powerful_model: str = os.getenv("GEMINI_POWERFUL_MODEL", "gemini-2.5-flash")

    ollama_model: str | None = os.getenv("OLLAMA_MODEL") or None
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))


settings = Settings()
