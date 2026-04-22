"""Canvas Analysis: 7 action fields, maturity grading, gaps & opportunities."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import CANVAS_SYSTEM, CANVAS_USER


class CanvasAnalysisAgent(BaseAgent):
    name = "canvas_analysis"
    task_type = "reasoning"
    complexity = "high"
    criticality = "high"
    system_prompt = CANVAS_SYSTEM
    user_template = CANVAS_USER
