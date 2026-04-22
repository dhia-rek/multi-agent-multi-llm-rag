"""Planner: reads the business case and produces a structured plan."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import PLANNER_SYSTEM, PLANNER_USER


class PlannerAgent(BaseAgent):
    name = "planner"
    task_type = "extraction"   # extract intent + structure
    complexity = "medium"
    criticality = "medium"
    system_prompt = PLANNER_SYSTEM
    user_template = PLANNER_USER
