"""Offline demo client: replays each agent's output from a recorded fixture.

Used when no GOOGLE_API_KEY is configured (DEMO_MODE=auto) or when forced with
DEMO_MODE=on. Retrieval still runs for real; only the LLM calls are replaced.
The fixture is recorded from one live Gemini run of the sample bikeshop case
(scripts/record_demo.py), so for any other input the agent outputs will still
describe the bikeshop. The UI says so.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from src.agents import prompts
from src.llm_clients.base import BaseLLMClient, LLMResponse
from src.utils.config import settings

# Agents are identified by their system prompt, which is unique per agent.
_AGENT_BY_SYSTEM = {
    prompts.PLANNER_SYSTEM: "planner",
    prompts.FRAMEWORK_SYSTEM: "framework",
    prompts.CANVAS_SYSTEM: "canvas",
    prompts.STRATEGIST_SYSTEM: "strategist",
    prompts.ROADMAP_SYSTEM: "roadmap",
    prompts.EVALUATOR_SYSTEM: "evaluator",
}


@cache
def load_fixture(path: Path) -> dict:
    """Read and memoise the recorded fixture JSON (agent name -> output)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DemoLLMClient(BaseLLMClient):
    """Serves canned agent outputs from the recorded fixture instead of calling an LLM."""

    def __init__(self, tier: str = "demo", fixture_path: Path | None = None) -> None:
        """Bind the client to a tier and fixture (defaults to `settings.demo_fixture`)."""
        self.name = f"demo-fixture-{tier}"
        self.tier = tier
        self.fixture_path = fixture_path or settings.demo_fixture

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        """Return the fixture output of the agent identified by `system`.

        Args:
            prompt: Ignored; the output is fixed per agent.
            system: System prompt used to identify the agent.
            **kwargs: Ignored.

        Returns:
            The agent's recorded output serialised as JSON text.

        Raises:
            KeyError: If the system prompt is unknown or the fixture lacks that agent.
        """
        agent = _AGENT_BY_SYSTEM.get(system or "")
        fixture = load_fixture(self.fixture_path)
        if agent is None or agent not in fixture:
            raise KeyError(f"No demo fixture output for this prompt (agent={agent!r})")
        return LLMResponse(text=json.dumps(fixture[agent]), model=self.name, tier=self.tier)
