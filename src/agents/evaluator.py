"""Evaluator: critiques the roadmap for coverage, consistency, and risk."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.prompts import EVALUATOR_SYSTEM, EVALUATOR_USER


class EvaluatorAgent(BaseAgent):
    name = "evaluator"
    task_type = "evaluation"
    complexity = "medium"
    criticality = "high"
    system_prompt = EVALUATOR_SYSTEM
    user_template = EVALUATOR_USER
