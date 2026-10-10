"""Common LLM client interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class QuotaExhaustedError(RuntimeError):
    """Raised when an LLM provider has permanently exhausted its quota for the day.
    Routers catch this and fail over to another tier (or to a mock)."""

    def __init__(self, model: str, message: str = ""):
        self.model = model
        super().__init__(message or f"Daily quota exhausted for {model}")


@dataclass
class LLMResponse:
    text: str
    model: str
    tier: str  # "fast" | "powerful" | "local" | "mock"
    usage: dict | None = None


class BaseLLMClient(ABC):
    name: str
    tier: str

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        ...

    def as_chat_model(self, thinking_budget: int | None = None):
        """This client as a LangChain chat model. Offline clients use the generic adapter;
        Gemini overrides this with a native ChatGoogleGenerativeAI."""
        from src.llm_clients.langchain_chat import ClientChatModel

        return ClientChatModel(client=self)
