"""Common agent scaffolding: retry, JSON-safe parsing, routing call."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.llm_clients.base import BaseLLMClient, QuotaExhaustedError
from src.llm_clients.gemini import invoke_with_retry, usage_from_message
from src.routing.router import Level, Router, RoutingDecision, TaskType
from src.utils.logging import get_logger

log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def parse_json_safely(raw: str) -> dict | list | str | int | float | bool | None:
    """Parse JSON from LLM output, tolerating code fences and surrounding chatter.

    Tries, in order: the content of a ```json fence (or the whole text), then the
    greedy span from the first ``{`` to the last ``}``.

    Args:
        raw: Raw model text.

    Returns:
        The decoded JSON value (including scalars), or None if the text is empty,
        invalid, or the JSON literal ``null``. Callers expecting structured output
        must check for a dict or list.
    """
    if not raw or not raw.strip():
        return None
    fenced = _FENCE.search(raw)
    candidate = fenced.group(1) if fenced else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(candidate)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


@dataclass
class AgentResult:
    """Outcome of one agent run.

    Attributes:
        name: Agent name.
        output: Decoded JSON value if parsing succeeded, else the raw text.
        raw_text: Stripped model response text.
        model: Name of the LLM client that answered.
        tier: Tier of that client.
        routing: Routing decision that selected the client.
        extras: Optional extra data; holds ``usage`` when token usage is available.
    """
    name: str
    output: dict | list | str | int | float | bool | None
    raw_text: str
    model: str
    tier: str
    routing: RoutingDecision
    extras: dict = field(default_factory=dict)


class BaseAgent:
    """Base class for LLM agents: subclasses set the class attributes below.

    Class attributes (``task_type``, ``complexity``, ``criticality``) feed the router;
    ``system_prompt`` / ``user_template`` define the prompt.
    """

    name: str = "base"
    task_type: TaskType = "synthesis"
    complexity: Level = "medium"
    criticality: Level = "medium"
    # Gemini thinking-token budget for this agent; None = tier default
    # (powerful tier: GEMINI_THINKING_BUDGET, fast tier: 0).
    thinking_budget: int | None = None
    system_prompt: str = ""
    user_template: str = ""

    def __init__(self, router: Router) -> None:
        """Store the router used to pick an LLM client for each call."""
        self.router = router

    def render_user(self, **kwargs: Any) -> str:
        """Fill ``user_template`` with ``kwargs`` via ``str.format``."""
        return self.user_template.format(**kwargs)

    def _messages(self, user_prompt: str) -> list[BaseMessage]:
        """Build the system + human message pair for a call."""
        return [SystemMessage(self.system_prompt), HumanMessage(user_prompt)]

    def _invoke(self, client: BaseLLMClient, user_prompt: str) -> BaseMessage:
        """Call ``client``'s chat model (with retry) using this agent's thinking budget."""
        return invoke_with_retry(client.as_chat_model(self.thinking_budget), self._messages(user_prompt))

    def call(self, user_prompt: str) -> tuple[BaseMessage, BaseLLMClient, RoutingDecision]:
        """Route, then call the tier's LangChain chat model.

        On a daily-quota error the tier is marked dead for the rest of the run and the
        call is re-routed (to a mock).

        Args:
            user_prompt: Rendered user message.

        Returns:
            Tuple of (LangChain message, client that answered, routing decision used).
        """
        client, decision = self.router.route(self.task_type, self.complexity, self.criticality)
        try:
            msg = self._invoke(client, user_prompt)
        except QuotaExhaustedError:
            self.router.mark_exhausted(decision.tier)
            client, decision = self.router.route(self.task_type, self.complexity, self.criticality)
            msg = self._invoke(client, user_prompt)
        return msg, client, decision

    def run(self, **inputs: Any) -> AgentResult:
        """Render the prompt, call the model and parse its JSON output.

        Args:
            **inputs: Values for the placeholders in ``user_template``.

        Returns:
            AgentResult; ``output`` falls back to the raw text (with a warning logged)
            when no JSON can be parsed.
        """
        user = self.render_user(**inputs)
        msg, client, decision = self.call(user)
        text = (msg.content if isinstance(msg.content, str) else str(msg.text)).strip()
        parsed = parse_json_safely(text)
        if parsed is None:
            log.warning("[%s] Could not parse JSON, returning raw text.", self.name)
        usage = usage_from_message(msg)
        return AgentResult(
            name=self.name,
            output=parsed if parsed is not None else text,
            raw_text=text,
            model=client.name,
            tier=client.tier,
            routing=decision,
            extras={"usage": usage} if usage else {},
        )
