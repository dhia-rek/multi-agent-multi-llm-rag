"""LangChain adapter for the non-Gemini clients (demo fixture, mock, Ollama).

Gemini is used natively through `langchain-google-genai`; everything the router
can answer with offline goes through this thin `BaseChatModel` wrapper, so the
agents and the graph only ever see one interface.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _text(msg: BaseMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


class ClientChatModel(BaseChatModel):
    """Expose a `BaseLLMClient` as a LangChain chat model."""

    client: Any

    @property
    def _llm_type(self) -> str:
        return f"client-{getattr(self.client, 'tier', 'unknown')}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n\n".join(_text(m) for m in messages if isinstance(m, SystemMessage)) or None
        user = "\n\n".join(_text(m) for m in messages if isinstance(m, HumanMessage))
        resp = self.client.generate(user, system=system)
        usage = None
        if resp.usage:
            p, o = resp.usage.get("prompt_tokens") or 0, resp.usage.get("output_tokens") or 0
            usage = {"input_tokens": p, "output_tokens": o, "total_tokens": p + o}
        msg = AIMessage(
            content=resp.text,
            usage_metadata=usage,
            response_metadata={"model_name": resp.model, "tier": resp.tier},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])
