"""Common LLM client interface: response container, base class and quota error."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class QuotaExhaustedError(RuntimeError):
    """Raised when an LLM provider has permanently exhausted its quota for the day.

    Routers catch this and fail over to another tier (or to a mock).

    Attributes:
        model: Name of the model whose quota ran out.
    """

    def __init__(self, model: str, message: str = "") -> None:
        """Store the model name and build a default message if none is given."""
        self.model = model
        super().__init__(message or f"Daily quota exhausted for {model}")


@dataclass
class LLMResponse:
    """Provider-independent result of one LLM call.

    Attributes:
        text: Generated text.
        model: Name of the model that produced it.
        tier: Router tier that served the call.
        usage: Token accounting (keys vary by client) or None when unavailable.
    """

    text: str
    model: str
    tier: str  # "fast" | "powerful" | "local" | "mock"
    usage: dict | None = None


class BaseLLMClient(ABC):
    """Interface every LLM backend (Gemini, Ollama, demo, mock) implements.

    Attributes:
        name: Model identifier reported in responses.
        tier: Router tier this client serves.
    """

    name: str
    tier: str

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            **kwargs: Client-specific options (e.g. temperature).

        Returns:
            The model's response.
        """

    def as_chat_model(self, thinking_budget: int | None = None) -> BaseChatModel:
        """This client as a LangChain chat model.

        Offline clients use the generic adapter; Gemini overrides this with a
        native ChatGoogleGenerativeAI.

        Args:
            thinking_budget: Thinking-token budget; ignored by this default
                implementation (only Gemini uses it).

        Returns:
            A LangChain chat model backed by this client.
        """
        from src.llm_clients.langchain_chat import ClientChatModel

        return ClientChatModel(client=self)
