"""Common agent scaffolding: retry, JSON-safe parsing, routing call."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.llm_clients.base import LLMResponse, QuotaExhaustedError
from src.routing.router import Router, RoutingDecision
from src.utils.logging import get_logger

log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def parse_json_safely(raw: str) -> dict | list | None:
    """LLMs sometimes wrap JSON in fences or add chatter. Be lenient."""
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
    name: str
    output: dict | list | str
    raw_text: str
    model: str
    tier: str
    routing: RoutingDecision
    extras: dict = field(default_factory=dict)


class BaseAgent:
    name: str = "base"
    task_type: str = "synthesis"
    complexity: str = "medium"
    criticality: str = "medium"
    system_prompt: str = ""
    user_template: str = ""

    def __init__(self, router: Router):
        self.router = router

    def render_user(self, **kwargs) -> str:
        return self.user_template.format(**kwargs)

    def call(self, user_prompt: str, **gen_kwargs) -> tuple[LLMResponse, RoutingDecision]:
        client, decision = self.router.route(self.task_type, self.complexity, self.criticality)
        try:
            resp = client.generate(user_prompt, system=self.system_prompt, **gen_kwargs)
        except QuotaExhaustedError:
            # Mark this tier dead for the rest of the run and re-route.
            self.router.mark_exhausted(decision.tier)
            client, decision = self.router.route(self.task_type, self.complexity, self.criticality)
            resp = client.generate(user_prompt, system=self.system_prompt, **gen_kwargs)
        return resp, decision

    def run(self, **inputs: Any) -> AgentResult:
        user = self.render_user(**inputs)
        resp, decision = self.call(user)
        parsed = parse_json_safely(resp.text)
        if parsed is None:
            log.warning("[%s] Could not parse JSON, returning raw text.", self.name)
        return AgentResult(
            name=self.name,
            output=parsed if parsed is not None else resp.text,
            raw_text=resp.text,
            model=resp.model,
            tier=resp.tier,
            routing=decision,
        )
