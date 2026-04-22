"""Deterministic mock client. Used as a fallback when no API key is configured,
so the pipeline can still be exercised end-to-end."""
from __future__ import annotations

from src.llm_clients.base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    def __init__(self, tier: str = "mock"):
        self.name = f"mock-{tier}"
        self.tier = tier

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        snippet = prompt[:300].replace("\n", " ")
        text = (
            "[MOCK OUTPUT — set GOOGLE_API_KEY in .env to get real generations]\n"
            f"role: {system[:80] if system else 'n/a'}\n"
            f"prompt-preview: {snippet}..."
        )
        return LLMResponse(text=text, model=self.name, tier=self.tier)
