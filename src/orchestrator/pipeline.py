"""End-to-end pipeline: business case in → structured roadmap + evaluation out.

The stages run as a LangGraph StateGraph (src/orchestrator/graph.py):
planner → retrieval → (framework ‖ canvas) → strategist → roadmap → evaluator.

Two run modes:
    .run(...)        → blocks until all stages are done, returns final trace.
    .run_streaming() → generator that yields after each stage; the UI uses this
                       to render live progress and per-agent annotations.
                       framework and canvas run in parallel, so they arrive in
                       completion order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.base import AgentResult
from src.cache.disk_cache import DiskCache
from src.orchestrator.graph import RESULT_FIELD, build_graph
from src.retrieval.retriever import RetrievedPassage, Retriever
from src.routing.router import Router
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


# graph node -> stream stage name (framework/canvas/... share their node name)
_STAGES = {*RESULT_FIELD, "retrieval"}
_ATTR = {**RESULT_FIELD, "retrieval": "retrieval"}


class Orchestrator:
    def __init__(self, router: Router | None = None, retriever: Retriever | None = None):
        self.router = router or Router()
        self.retriever = retriever or Retriever()
        self.graph = build_graph(self.router, self.retriever)
        self.cache = DiskCache("pipeline_v2")  # v2: LangGraph orchestration + reworked prompts; old traces are not reused

    def run(self, business_case: str, *, run_evaluator: bool = True) -> PipelineTrace:
        """Drain the streaming run and return the final trace."""
        trace = None
        for _stage, trace in self.run_streaming(business_case, run_evaluator=run_evaluator):  # noqa: B007
            pass
        assert trace is not None
        return trace

    def run_streaming(self, business_case: str, *, run_evaluator: bool = True):
        """Yield (stage_name, trace_so_far) after each stage completes.
        Lets the UI render progressively. Final yield is ('done', trace)."""
        cache_key = {
            "business_case": business_case,
            "evaluator": run_evaluator,
            "models": self.router.signature(),
        }
        cached = self.cache.get(cache_key)
        if cached is not None:
            log.info("Cache HIT for this business case — returning cached trace.")
            yield "cached", cached
            yield "done", cached
            return

        trace = PipelineTrace(business_case=business_case)
        inputs = {"business_case": business_case, "run_evaluator": run_evaluator}
        for update in self.graph.stream(inputs, stream_mode="updates"):
            for node, delta in update.items():
                if node not in _STAGES or not delta or _ATTR[node] not in delta:
                    continue  # join node, or an agent that is being retried
                setattr(trace, _ATTR[node], delta[_ATTR[node]])
                trace.timings.update(delta.get("timings", {}))
                trace.model_per_stage.update(delta.get("model_per_stage", {}))
                yield node, trace

        agents = [trace.plan, trace.framework, trace.canvas, trace.strategy, trace.roadmap, trace.evaluation]
        unparsed = [a.name for a in agents if a is not None and not isinstance(a.output, (dict, list))]
        if self.router.fallback_used:
            log.warning("A tier fell back to mock during this run; not caching the trace.")
        elif unparsed:
            log.warning("No valid JSON from %s; not caching the trace.", ", ".join(unparsed))
        else:
            self.cache.set(cache_key, trace)
        yield "done", trace
