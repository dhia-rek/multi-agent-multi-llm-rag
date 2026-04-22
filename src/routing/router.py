"""Multi-LLM router. Picks a client per task based on explicit criteria.

Routing policy
--------------
- task_type ∈ {extraction, synthesis, reasoning, generation, evaluation}
- complexity ∈ {low, medium, high}
- criticality ∈ {low, medium, high}

Rule of thumb:
    fast tier (Gemini Flash)  : extraction / reformulation / first drafts
    powerful tier (Gemini Pro): strategic reasoning, final generation,
                                roadmap synthesis, evaluation
    local tier (Ollama)       : optional, repetitive / experimental
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.llm_clients.base import BaseLLMClient, QuotaExhaustedError
from src.llm_clients.gemini import GeminiClient
from src.llm_clients.mock import MockLLMClient
from src.llm_clients.ollama import OllamaClient
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

TaskType = Literal["extraction", "synthesis", "reasoning", "generation", "evaluation"]
Level = Literal["low", "medium", "high"]


@dataclass
class RoutingDecision:
    tier: str
    reason: str


class Router:
    """Lazily instantiates one client per tier and serves them on demand."""

    def __init__(self):
        self._fast: BaseLLMClient | None = None
        self._powerful: BaseLLMClient | None = None
        self._local: BaseLLMClient | None = None
        self._has_gemini = bool(settings.google_api_key)
        self._has_ollama = bool(settings.ollama_model)
        # Tiers that hit a permanent (daily) quota cap during this run.
        self._dead_tiers: set[str] = set()
        if not self._has_gemini:
            log.warning(
                "GOOGLE_API_KEY missing — falling back to MockLLMClient. "
                "Set the key in .env for real generations."
            )

    def mark_exhausted(self, tier: str) -> None:
        """Called by callers when a tier returns a daily-quota error.
        We swap that tier's client for a Mock so the pipeline still completes."""
        if tier not in self._dead_tiers:
            log.warning("Tier '%s' marked exhausted; falling back to mock.", tier)
        self._dead_tiers.add(tier)
        if tier == "fast":
            self._fast = MockLLMClient(tier="fast")
        elif tier == "powerful":
            self._powerful = MockLLMClient(tier="powerful")
        elif tier == "local":
            self._local = MockLLMClient(tier="local")

    def fast(self) -> BaseLLMClient:
        if self._fast is None:
            self._fast = (
                GeminiClient(settings.gemini_fast_model, tier="fast")
                if self._has_gemini and "fast" not in self._dead_tiers
                else MockLLMClient(tier="fast")
            )
        return self._fast

    def powerful(self) -> BaseLLMClient:
        if self._powerful is None:
            self._powerful = (
                GeminiClient(settings.gemini_powerful_model, tier="powerful")
                if self._has_gemini and "powerful" not in self._dead_tiers
                else MockLLMClient(tier="powerful")
            )
        return self._powerful

    def local(self) -> BaseLLMClient | None:
        if not self._has_ollama:
            return None
        if self._local is None:
            self._local = OllamaClient()
        return self._local

    def decide(
        self,
        task_type: TaskType,
        complexity: Level = "medium",
        criticality: Level = "medium",
    ) -> RoutingDecision:
        # Critical or complex strategy/generation/eval → powerful
        if task_type in {"reasoning", "generation", "evaluation"} or complexity == "high" or criticality == "high":
            return RoutingDecision(
                tier="powerful",
                reason=f"task={task_type}, complexity={complexity}, criticality={criticality} → powerful tier",
            )
        # Repetitive / low-criticality with a local model available → local
        if criticality == "low" and self._has_ollama:
            return RoutingDecision(
                tier="local",
                reason=f"task={task_type}, criticality=low, local model available → local tier",
            )
        return RoutingDecision(
            tier="fast",
            reason=f"task={task_type}, complexity={complexity} → fast tier",
        )

    def route(
        self,
        task_type: TaskType,
        complexity: Level = "medium",
        criticality: Level = "medium",
    ) -> tuple[BaseLLMClient, RoutingDecision]:
        decision = self.decide(task_type, complexity, criticality)
        client = {
            "fast": self.fast(),
            "powerful": self.powerful(),
            "local": self.local() or self.fast(),
        }[decision.tier]
        log.info("Routing → %s (%s)", client.name, decision.reason)
        return client, decision
