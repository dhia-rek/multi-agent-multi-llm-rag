"""Optional Ollama client for a local open-source model. Disabled if OLLAMA_MODEL is empty."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from src.llm_clients.base import BaseLLMClient, LLMResponse
from src.utils.config import settings


class OllamaClient(BaseLLMClient):
    """Client for a local Ollama daemon (tier "local") via its HTTP /api/generate."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        """Resolve model (arg, then settings, then "llama3.2") and host (arg, then settings)."""
        self.name = model or settings.ollama_model or "llama3.2"
        self.tier = "local"
        self.host = host or settings.ollama_host

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        """Call Ollama non-streaming.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            **kwargs: Supports `temperature` (default 0.3) and
                `max_output_tokens` (default 2048, sent as num_predict).

        Returns:
            The stripped response text.

        Raises:
            RuntimeError: If the daemon cannot be reached.
        """
        body = {
            "model": self.name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.3),
                "num_predict": kwargs.get("max_output_tokens", 2048),
            },
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama unreachable at {self.host}. Is the daemon running?"
            ) from e
        return LLMResponse(text=data.get("response", "").strip(), model=self.name, tier=self.tier)
