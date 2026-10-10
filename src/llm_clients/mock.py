"""Deterministic mock client.

Used as a fallback when no API key is configured, so the pipeline can still be
exercised end-to-end.
"""
from __future__ import annotations

from typing import Any

from src.llm_clients.base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Offline client that echoes a preview of the prompt, with no model call."""

    def __init__(self, tier: str = "mock") -> None:
        """Create a mock serving the given tier name."""
        self.name = f"mock-{tier}"
        self.tier = tier

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        """Return a placeholder text with truncated role and prompt previews.

        Args:
            prompt: Prompt; the first 300 characters are echoed.
            system: System prompt; the first 80 characters are echoed.
            **kwargs: Ignored.

        Returns:
            A response flagged as mock output.
        """
        snippet = prompt[:300].replace("\n", " ")
        text = (
            "[MOCK OUTPUT — set GOOGLE_API_KEY in .env to get real generations]\n"
            f"role: {system[:80] if system else 'n/a'}\n"
            f"prompt-preview: {snippet}..."
        )
        return LLMResponse(text=text, model=self.name, tier=self.tier)
