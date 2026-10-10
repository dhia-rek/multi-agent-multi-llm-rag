"""Roadmap Generator: turns strategy into a structured, time-phased roadmap."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import ROADMAP_SYSTEM, ROADMAP_USER


class RoadmapAgent(BaseAgent):
    """Agent that produces the time-phased roadmap; thinking is disabled (budget 0) since it only reshapes a finished strategy."""

    name = "roadmap"
    task_type = "generation"
    complexity = "high"
    criticality = "high"
    thinking_budget = 0   # structured JSON from a finished strategy: little to think about
    system_prompt = ROADMAP_SYSTEM
    user_template = ROADMAP_USER
