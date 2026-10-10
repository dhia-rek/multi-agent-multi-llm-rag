"""End-to-end pipeline: business case in → structured roadmap + evaluation out.

Pipeline stages (with conditional execution and caching):
    1. Planner          (Flash)   — extract intent, build retrieval queries
    2. Retrieve         (RAG)     — pull passages from FAISS
    3. Framework        (Flash)   — classify by Why/What/How + 7 fields
    4. Canvas Analysis  (Pro)     — score maturity, find gaps
    5. Strategist       (Pro)     — synthesize strategic trajectory
    6. Roadmap          (Pro)     — produce structured roadmap
    7. Evaluator        (Pro)     — critique result

Two run modes:
    .run(...)        → blocks until all stages are done, returns final trace.
    .run_streaming() → generator that yields after each stage; the UI uses this
                       to render live progress and per-agent annotations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import AgentResult
from src.agents.canvas_analysis import CanvasAnalysisAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.framework import FrameworkAgent
from src.agents.planner import PlannerAgent
from src.agents.roadmap import RoadmapAgent
from src.agents.strategist import StrategistAgent
from src.cache.disk_cache import DiskCache
from src.retrieval.retriever import Retriever, RetrievedPassage
from src.routing.router import Router
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PipelineTrace:
    business_case: str
    plan: AgentResult | None = None
    retrieval: list[RetrievedPassage] = field(default_factory=list)
    framework: AgentResult | None = None
    canvas: AgentResult | None = None
    strategy: AgentResult | None = None
    roadmap: AgentResult | None = None
    evaluation: AgentResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    model_per_stage: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        def _agent(a: AgentResult | None) -> dict | None:
            if a is None:
                return None
            return {
                "output": a.output,
                "model": a.model,
                "tier": a.tier,
                "routing_reason": a.routing.reason,
            }

        return {
            "plan": _agent(self.plan),
            "retrieval": [
                {
                    "framework": p.chunk.framework,
                    "title": p.chunk.title,
                    "page": p.chunk.page,
                    "score": round(p.score, 4),
                    "preview": p.chunk.text[:300],
                }
                for p in self.retrieval
            ],
            "framework": _agent(self.framework),
            "canvas": _agent(self.canvas),
            "strategy": _agent(self.strategy),
            "roadmap": _agent(self.roadmap),
            "evaluation": _agent(self.evaluation),
            "timings": self.timings,
            "model_per_stage": self.model_per_stage,
        }


def _as_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


class Orchestrator:
    def __init__(self, router: Router | None = None, retriever: Retriever | None = None):
        self.router = router or Router()
        self.retriever = retriever or Retriever()
        self.planner = PlannerAgent(self.router)
        self.framework = FrameworkAgent(self.router)
        self.canvas = CanvasAnalysisAgent(self.router)
        self.strategist = StrategistAgent(self.router)
        self.roadmap = RoadmapAgent(self.router)
        self.evaluator = EvaluatorAgent(self.router)
        self.cache = DiskCache("pipeline")

    def _retrieve(self, queries: list[str], k_per_query: int = 4) -> list[RetrievedPassage]:
        seen = set()
        merged: list[RetrievedPassage] = []
        for q in queries:
            for p in self.retriever.retrieve(q, k=k_per_query):
                key = (p.chunk.chunk_id,)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(p)
        merged.sort(key=lambda p: -p.score)
        return merged[: settings.top_k * 2]

    def run(self, business_case: str, *, run_evaluator: bool = True) -> PipelineTrace:
        """Drain the streaming run and return the final trace."""
        trace = None
        for _stage, trace in self.run_streaming(business_case, run_evaluator=run_evaluator):
            pass
        assert trace is not None
        return trace

    def run_streaming(self, business_case: str, *, run_evaluator: bool = True):
        """Yield (stage_name, trace_so_far) after each stage completes.
        Lets the UI render progressively. Final yield is ('done', trace)."""
        cache_key = {"business_case": business_case, "evaluator": run_evaluator}
        cached = self.cache.get(cache_key)
        if cached is not None:
            log.info("Cache HIT for this business case — returning cached trace.")
            yield "cached", cached
            yield "done", cached
            return

        trace = PipelineTrace(business_case=business_case)

        # 1. Planner
        t = time.time()
        trace.plan = self.planner.run(business_case=business_case)
        trace.timings["planner"] = round(time.time() - t, 2)
        trace.model_per_stage["planner"] = trace.plan.model
        yield "planner", trace

        plan_obj = trace.plan.output if isinstance(trace.plan.output, dict) else {}
        queries = plan_obj.get("retrieval_queries") or [business_case[:300]]

        # 2. Retrieval
        t = time.time()
        trace.retrieval = self._retrieve(queries)
        trace.timings["retrieval"] = round(time.time() - t, 2)
        context = Retriever.format_context(trace.retrieval)
        yield "retrieval", trace

        # 3. Framework agent
        t = time.time()
        trace.framework = self.framework.run(
            company_summary=plan_obj.get("company_summary", "")[:600],
            context=context[:6000],
        )
        trace.timings["framework"] = round(time.time() - t, 2)
        trace.model_per_stage["framework"] = trace.framework.model
        yield "framework", trace

        # 4. Canvas analysis
        t = time.time()
        trace.canvas = self.canvas.run(
            company_summary=plan_obj.get("company_summary", "")[:600],
            objectives=_as_text(plan_obj.get("objectives", [])),
            constraints=_as_text(plan_obj.get("constraints", [])),
            framework_insights=_as_text(trace.framework.output)[:4000],
            context=context[:4000],
        )
        trace.timings["canvas"] = round(time.time() - t, 2)
        trace.model_per_stage["canvas"] = trace.canvas.model
        yield "canvas", trace

        # 5. Strategist
        t = time.time()
        trace.strategy = self.strategist.run(
            plan=_as_text(plan_obj)[:3000],
            canvas=_as_text(trace.canvas.output)[:3000],
            framework_insights=_as_text(trace.framework.output)[:3000],
        )
        trace.timings["strategist"] = round(time.time() - t, 2)
        trace.model_per_stage["strategist"] = trace.strategy.model
        yield "strategist", trace

        # 6. Roadmap
        t = time.time()
        trace.roadmap = self.roadmap.run(
            strategy=_as_text(trace.strategy.output)[:4000],
            canvas=_as_text(trace.canvas.output)[:3000],
        )
        trace.timings["roadmap"] = round(time.time() - t, 2)
        trace.model_per_stage["roadmap"] = trace.roadmap.model
        yield "roadmap", trace

        # 7. Evaluator (optional, conditional execution)
        if run_evaluator:
            t = time.time()
            trace.evaluation = self.evaluator.run(
                business_case=business_case[:2000],
                plan=_as_text(plan_obj)[:2000],
                strategy=_as_text(trace.strategy.output)[:2000],
                roadmap=_as_text(trace.roadmap.output)[:4000],
            )
            trace.timings["evaluator"] = round(time.time() - t, 2)
            trace.model_per_stage["evaluator"] = trace.evaluation.model
            yield "evaluator", trace

        self.cache.set(cache_key, trace)
        yield "done", trace
