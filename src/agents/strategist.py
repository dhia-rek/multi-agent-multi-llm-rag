"""Strategist: synthesizes purpose, pillars, expected value, risks, initiatives."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import STRATEGIST_SYSTEM, STRATEGIST_USER


class StrategistAgent(BaseAgent):
    name = "strategist"
    task_type = "reasoning"
    complexity = "high"
    criticality = "high"
    system_prompt = STRATEGIST_SYSTEM
    user_template = STRATEGIST_USER
