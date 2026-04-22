"""Framework agent: classifies retrieved passages by Why/What/How + 7 fields."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import FRAMEWORK_SYSTEM, FRAMEWORK_USER


class FrameworkAgent(BaseAgent):
    name = "framework"
    task_type = "synthesis"
    complexity = "medium"
    criticality = "medium"
    system_prompt = FRAMEWORK_SYSTEM
    user_template = FRAMEWORK_USER
