"""Roadmap Generator: turns strategy into a structured, time-phased roadmap."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import ROADMAP_SYSTEM, ROADMAP_USER


class RoadmapAgent(BaseAgent):
    name = "roadmap"
    task_type = "generation"
    complexity = "high"
    criticality = "high"
    system_prompt = ROADMAP_SYSTEM
    user_template = ROADMAP_USER
