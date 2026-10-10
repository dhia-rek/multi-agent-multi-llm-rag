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
    """Immutable app settings.

    Env-backed defaults are read once at import time (class-body evaluation), so
    changing the environment afterwards has no effect on the ``settings`` singleton.
    """

    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    index_dir: Path = PROJECT_ROOT / "vector_store" / "index"
    cache_dir: Path = PROJECT_ROOT / ".cache"

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    gemini_fast_model: str = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")
    gemini_powerful_model: str = os.getenv("GEMINI_POWERFUL_MODEL", "gemini-2.5-flash")
    # Gemini 2.5 thinking tokens count against max_output_tokens. Unbounded, they
    # ate ~1960 of 2048 and truncated every powerful-tier JSON. The powerful tier
    # gets this budget on top of the answer budget; the fast tier doesn't think.
    gemini_thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "1024"))

    ollama_model: str | None = os.getenv("OLLAMA_MODEL") or None
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))

    # Offline demo: "auto" = replay the recorded fixture outputs when no Gemini key is set,
    # "on" = always, "off" = never (falls back to the placeholder MockLLMClient).
    demo_mode: str = os.getenv("DEMO_MODE", "auto").strip().lower()
    demo_fixture: Path = PROJECT_ROOT / "data" / "demo" / "alexandra_bikeshop.fixture.json"

    @property
    def demo_active(self) -> bool:
        """Whether to replay recorded fixture outputs instead of calling a live LLM.

        True for ``DEMO_MODE`` on/1/true, False for off/0/false; otherwise ("auto")
        True only when no Google API key is configured.
        """
        if self.demo_mode in {"on", "1", "true"}:
            return True
        if self.demo_mode in {"off", "0", "false"}:
            return False
        return not self.google_api_key


settings = Settings()
