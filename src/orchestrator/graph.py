"""The pipeline as a LangGraph `StateGraph`.

    START ─ planner ─ retrieval ─┬─ framework ─┐
                                 └─ canvas ────┴─ join ─ strategist ─ roadmap ─┬─ evaluator ─ END
                                                                               └─ END (run_evaluator=False)

Real data dependencies (what each agent's prompt actually consumes):

    planner     business case
    retrieval   planner's queries
    framework   planner summary + retrieved context
    canvas      planner summary/objectives/constraints + retrieved context   (no framework output)
    strategist  plan + canvas + framework
    roadmap     strategy + canvas
    evaluator   business case + plan + strategy + roadmap

So the chain is sequential from strategist on, and only one pair runs side by side:
framework ‖ canvas (the canvas prompt no longer reads framework's output).

Retries/fallbacks are conditional edges: an agent that returns unparsable JSON from a live
model loops back to itself once. Quota exhaustion fails over to a mock inside the agent
call (`BaseAgent.call`), and the planner's missing queries fall back to the raw case text.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base import AgentResult, BaseAgent
from src.agents.canvas_analysis import CanvasAnalysisAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.framework import FrameworkAgent
from src.agents.planner import PlannerAgent
from src.agents.roadmap import RoadmapAgent
from src.agents.strategist import StrategistAgent
from src.retrieval.retriever import RetrievedPassage, Retriever
from src.routing.router import Router
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

MAX_RETRIES = 1  # extra attempts after an unparsable answer from a live model


def merge(a: dict | None, b: dict | None) -> dict:
    """State reducer: shallow-merge dicts, later keys win (parallel branches write distinct keys)."""
    return {**(a or {}), **(b or {})}


def add(a: dict | None, b: dict | None) -> dict:
    """State reducer: sum numeric values per key, so retries/time accumulate across attempts."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class PipelineState(TypedDict, total=False):
    """LangGraph state.

    Each agent result lands in its own field once parsed;
    a missing field is what makes a conditional edge loop back to retry.

    `timings` (seconds per stage), `model_per_stage`, `retries` (extra retry counts)
    and `spent` (seconds summed over attempts) use reducers so concurrent
    branches can update them without overwriting each other.
    """

    business_case: str
    run_evaluator: bool
    plan: AgentResult
    retrieval: list[RetrievedPassage]
    context: str
    framework: AgentResult
    canvas: AgentResult
    strategy: AgentResult
    roadmap: AgentResult
    evaluation: AgentResult
    # Reducers: parallel branches write different keys of these dicts.
    timings: Annotated[dict, merge]
    model_per_stage: Annotated[dict, merge]
    retries: Annotated[dict, add]
    spent: Annotated[dict, add]  # seconds, summed over attempts


def as_text(value: Any) -> str:
    """Compact JSON: ~25% fewer prompt tokens than indent=2, same information."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def is_parsed(res: AgentResult) -> bool:
    """True if the agent output was parsed into JSON (dict or list), not left as raw text."""
    return isinstance(res.output, (dict, list))


def is_offline(res: AgentResult) -> bool:
    """True if a mock or demo client produced the result (retrying would not help)."""
    return res.model.startswith(("mock-", "demo-"))


def _plan_obj(state: PipelineState) -> dict:
    """The planner's parsed output, or {} if absent or unparsed."""
    plan = state.get("plan")
    return plan.output if plan is not None and isinstance(plan.output, dict) else {}


def build_inputs(stage: str, s: PipelineState) -> dict[str, str]:
    """Prompt variables per agent, trimmed to what each prompt needs.

    Args:
        stage: Agent name (planner, framework, canvas, strategist, roadmap, evaluator).
        s: Current pipeline state; upstream results must already be present.

    Returns:
        Prompt variable name -> text, truncated by character limits.

    Raises:
        KeyError: For an unknown stage, or if a required upstream result is missing.
    """
    plan = _plan_obj(s)
    ctx = s.get("context", "")
    if stage == "planner":
        return {"business_case": s["business_case"]}
    if stage == "framework":
        return {"company_summary": plan.get("company_summary", "")[:600], "context": ctx[:6000]}
    if stage == "canvas":
        return {
            "company_summary": plan.get("company_summary", "")[:600],
            "objectives": as_text(plan.get("objectives", [])),
            "constraints": as_text(plan.get("constraints", [])),
            "context": ctx[:4000],
        }
    if stage == "strategist":
        return {
            "plan": as_text(plan)[:3000],
            "canvas": as_text(s["canvas"].output)[:3000],
            "framework_insights": as_text(s["framework"].output)[:3000],
        }
    if stage == "roadmap":
        return {"strategy": as_text(s["strategy"].output)[:4000], "canvas": as_text(s["canvas"].output)[:3000]}
    if stage == "evaluator":
        return {
            "business_case": s["business_case"][:2000],
            "plan": as_text(plan)[:2000],
            "strategy": as_text(s["strategy"].output)[:2000],
            "roadmap": as_text(s["roadmap"].output)[:4000],
        }
    raise KeyError(stage)


# stage -> state field holding its AgentResult
RESULT_FIELD = {
    "planner": "plan",
    "framework": "framework",
    "canvas": "canvas",
    "strategist": "strategy",
    "roadmap": "roadmap",
    "evaluator": "evaluation",
}


def merge_retrieval(per_query: list[list[RetrievedPassage]]) -> list[RetrievedPassage]:
    """Keep the first passage for each chunk id, then sort retained passages by score.

    Args:
        per_query: One passage list per retrieval query.

    Returns:
        At most `settings.top_k * 2` passages sorted by descending score.
    """
    seen: set[str] = set()
    merged: list[RetrievedPassage] = []
    for passages in per_query:
        for p in passages:
            if p.chunk.chunk_id in seen:
                continue
            seen.add(p.chunk.chunk_id)
            merged.append(p)
    merged.sort(key=lambda p: -p.score)
    return merged[: settings.top_k * 2]


def build_graph(router: Router, retriever: Retriever) -> CompiledStateGraph:
    """Compile the pipeline graph for this router/retriever.

    Args:
        router: Supplies the LLM client for each agent.
        retriever: Used by the retrieval node.

    Returns:
        The compiled graph; stream it with `{"business_case": ..., "run_evaluator": ...}`.
    """
    agents: dict[str, BaseAgent] = {
        "planner": PlannerAgent(router),
        "framework": FrameworkAgent(router),
        "canvas": CanvasAnalysisAgent(router),
        "strategist": StrategistAgent(router),
        "roadmap": RoadmapAgent(router),
        "evaluator": EvaluatorAgent(router),
    }

    def agent_node(stage: str) -> Callable[[PipelineState], dict[str, Any]]:
        """Node factory: run one agent.

        On unparsable live output it returns only retry counters (no result
        field) so the conditional edge loops back.
        """
        agent, field = agents[stage], RESULT_FIELD[stage]

        def node(state: PipelineState) -> dict[str, Any]:
            """Run the agent once; return its result, or retry counters if unparsable."""
            attempt = state.get("retries", {}).get(stage, 0)
            t = time.time()
            res = agent.run(**build_inputs(stage, state))
            took = time.time() - t
            if not is_parsed(res) and not is_offline(res) and attempt < MAX_RETRIES:
                log.warning("[%s] unparsable output, retrying (%d/%d).", stage, attempt + 1, MAX_RETRIES)
                return {"retries": {stage: 1}, "spent": {stage: took}}  # no result → edge loops back
            spent = state.get("spent", {}).get(stage, 0.0) + took
            return {
                field: res,
                "timings": {stage: round(spent, 2)},
                "model_per_stage": {stage: res.model},
            }

        return node

    def retrieval(state: PipelineState) -> dict:
        """Run the planner's queries (raw case text if it gave none) and merge the passages."""
        t = time.time()
        plan = _plan_obj(state)
        queries = plan.get("retrieval_queries")
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in queries or [] if isinstance(q, str) and q.strip()] or [state["business_case"][:300]]
        passages = merge_retrieval(retriever.retrieve_many(queries, k=4))
        return {
            "retrieval": passages,
            "context": Retriever.format_context(passages),
            "timings": {"retrieval": round(time.time() - t, 2)},
        }

    def join(state: PipelineState) -> dict:
        """No-op barrier node after framework and canvas; routing happens in `join_edge`."""
        return {}

    def retry_edge(stage: str, nxt: str) -> Callable[[PipelineState], str]:
        """Edge router: go to `nxt` once the stage has a result, else back to the stage."""
        field = RESULT_FIELD[stage]
        return lambda s: nxt if s.get(field) is not None else stage

    def join_edge(s: PipelineState) -> str:
        """Route after join: strategist if both parallel branches are done, else END."""
        # Fires once per finished branch; only the last one to arrive moves on.
        return "strategist" if s.get("framework") is not None and s.get("canvas") is not None else END

    def eval_edge(s: PipelineState) -> str:
        """Route to the evaluator unless `run_evaluator` is False."""
        return "evaluator" if s.get("run_evaluator", True) else END

    def roadmap_edge(s: PipelineState) -> str:
        """Retry roadmap if it has no result, else continue via `eval_edge`."""
        if s.get("roadmap") is None:
            return "roadmap"  # unparsable output: retry
        return eval_edge(s)

    g = StateGraph(PipelineState)
    for stage in agents:
        g.add_node(stage, agent_node(stage))  # type: ignore[call-overload]  # LangGraph stubs reject a TypedDict node factory
    g.add_node("retrieval", retrieval)
    g.add_node("join", join)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", retry_edge("planner", "retrieval"), ["planner", "retrieval"])
    g.add_edge("retrieval", "framework")
    g.add_edge("retrieval", "canvas")
    g.add_conditional_edges("framework", retry_edge("framework", "join"), ["framework", "join"])
    g.add_conditional_edges("canvas", retry_edge("canvas", "join"), ["canvas", "join"])
    g.add_conditional_edges("join", join_edge, ["strategist", END])
    g.add_conditional_edges("strategist", retry_edge("strategist", "roadmap"), ["strategist", "roadmap"])
    g.add_conditional_edges("roadmap", roadmap_edge, ["roadmap", "evaluator", END])
    g.add_conditional_edges("evaluator", retry_edge("evaluator", END), ["evaluator", END])
    return g.compile()
