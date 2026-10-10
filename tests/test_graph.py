"""Graph structure, parallel branches, retries and fallbacks (all offline)."""
import threading
import time

from src.agents.base import AgentResult
from src.llm_clients.base import QuotaExhaustedError
from src.orchestrator import graph as graph_mod
from src.orchestrator.graph import build_inputs, is_parsed
from src.routing.router import RoutingDecision

CASE = "Alexandra runs a small bike shop with two stores and wants to digitize repairs."


def _edges(orch):
    g = orch.graph.get_graph()
    return {(e.source, e.target) for e in g.edges}


def test_graph_has_one_node_per_agent(orchestrator):
    nodes = set(orchestrator.graph.get_graph().nodes)
    assert {"planner", "retrieval", "framework", "canvas", "join", "strategist", "roadmap", "evaluator"} <= nodes


def test_framework_and_canvas_fan_out_from_retrieval_and_join(orchestrator):
    e = _edges(orchestrator)
    assert ("retrieval", "framework") in e and ("retrieval", "canvas") in e
    assert ("framework", "join") in e and ("canvas", "join") in e
    # the chain after the join is strictly sequential
    assert ("join", "strategist") in e and ("strategist", "roadmap") in e and ("roadmap", "evaluator") in e


def test_canvas_prompt_does_not_depend_on_framework():
    from src.agents.prompts import CANVAS_USER

    assert "framework_insights" not in CANVAS_USER
    state = {"business_case": CASE, "context": "ctx"}
    assert "framework_insights" not in build_inputs("canvas", state)


def test_framework_and_canvas_run_concurrently(orchestrator, monkeypatch):
    barrier = threading.Barrier(2, timeout=5)  # both must be inside run() at once, or this times out
    from src.agents.canvas_analysis import CanvasAnalysisAgent
    from src.agents.framework import FrameworkAgent

    for cls in (FrameworkAgent, CanvasAnalysisAgent):
        orig = cls.run

        def slow(self, _orig=orig, **kw):
            barrier.wait()
            return _orig(self, **kw)

        monkeypatch.setattr(cls, "run", slow)
    trace = orchestrator.run(CASE)
    assert trace.framework and trace.canvas and trace.roadmap


def test_parallel_stage_timings_overlap(orchestrator, monkeypatch):
    from src.agents.canvas_analysis import CanvasAnalysisAgent
    from src.agents.framework import FrameworkAgent

    for cls in (FrameworkAgent, CanvasAnalysisAgent):
        orig = cls.run
        monkeypatch.setattr(cls, "run", lambda self, _o=orig, **kw: (time.sleep(0.4), _o(self, **kw))[1])
    t = time.time()
    orchestrator.run(CASE, run_evaluator=False)
    assert time.time() - t < 0.75  # two 0.4s stages in parallel, not 0.8s


def test_demo_mode_trace_shape(orchestrator):
    trace = orchestrator.run(CASE)
    assert all(is_parsed(a) for a in (trace.plan, trace.framework, trace.canvas, trace.strategy, trace.roadmap))
    assert trace.retrieval and trace.timings["retrieval"] >= 0


def _result(text, model="gemini-x"):
    return AgentResult("x", text, "", model, "fast", RoutingDecision("fast", "t"))


def test_unparsable_live_output_is_retried_once(orchestrator, monkeypatch):
    from src.agents.planner import PlannerAgent

    orig, calls = PlannerAgent.run, []

    def flaky(self, **kw):
        calls.append(1)
        if len(calls) == 1:
            return _result('{"cut off')
        return orig(self, **kw)

    monkeypatch.setattr(PlannerAgent, "run", flaky)
    trace = orchestrator.run(CASE, run_evaluator=False)
    assert len(calls) == 2
    assert isinstance(trace.plan.output, dict)


def test_retry_is_bounded(orchestrator, monkeypatch):
    from src.agents.planner import PlannerAgent

    calls = []
    monkeypatch.setattr(PlannerAgent, "run", lambda self, **kw: calls.append(1) or _result("never json"))
    trace = orchestrator.run(CASE, run_evaluator=False)
    assert len(calls) == 1 + graph_mod.MAX_RETRIES
    assert trace.plan.output == "never json"  # kept, pipeline still finishes
    assert trace.roadmap is not None


def test_mock_output_is_not_retried(orchestrator, monkeypatch):
    from src.agents.planner import PlannerAgent

    calls = []
    monkeypatch.setattr(
        PlannerAgent, "run", lambda self, **kw: calls.append(1) or _result("mock text", "mock-fast")
    )
    orchestrator.run(CASE, run_evaluator=False)
    assert len(calls) == 1


def test_missing_planner_queries_fall_back_to_case_text(orchestrator, monkeypatch):
    from src.agents.planner import PlannerAgent

    monkeypatch.setattr(PlannerAgent, "run", lambda self, **kw: _result({"company_summary": "x"}, "demo-fixture"))
    seen = []
    real = orchestrator.retriever.retrieve_many
    monkeypatch.setattr(orchestrator.retriever, "retrieve_many", lambda q, k=None: seen.append(q) or real(q, k))
    orchestrator.run(CASE, run_evaluator=False)
    assert seen == [[CASE[:300]]]


def test_quota_exhaustion_fails_over_to_mock_and_finishes(orchestrator, monkeypatch):
    from src.agents.base import BaseAgent

    orig = BaseAgent._invoke
    fired = []

    def invoke(self, client, prompt):
        if client.tier == "powerful" and not fired:
            fired.append(1)
            raise QuotaExhaustedError("gemini-x")
        return orig(self, client, prompt)

    monkeypatch.setattr(BaseAgent, "_invoke", invoke)
    trace = orchestrator.run(CASE)
    assert orchestrator.router.fallback_used
    assert trace.roadmap is not None and trace.evaluation is not None


def test_evaluator_skipped_edge(orchestrator):
    stages = [s for s, _ in orchestrator.run_streaming(CASE, run_evaluator=False)]
    assert "evaluator" not in stages and stages[-1] == "done"


def test_retrieve_many_matches_single_queries(fake_retriever):
    queries = ["online booking", "staff training"]
    batch = fake_retriever.retrieve_many(queries, k=2)
    for q, got in zip(queries, batch, strict=True):
        assert [p.chunk.chunk_id for p in got] == [p.chunk.chunk_id for p in fake_retriever.retrieve(q, k=2)]


def test_strategist_runs_exactly_once_under_random_branch_timing(orchestrator, monkeypatch):
    import random

    from src.agents.canvas_analysis import CanvasAnalysisAgent
    from src.agents.framework import FrameworkAgent
    from src.agents.strategist import StrategistAgent

    count = []
    orig_s = StrategistAgent.run
    monkeypatch.setattr(StrategistAgent, "run", lambda self, **kw: count.append(1) or orig_s(self, **kw))
    for cls in (FrameworkAgent, CanvasAnalysisAgent):
        orig = cls.run
        monkeypatch.setattr(
            cls, "run", lambda self, _o=orig, **kw: (time.sleep(random.random() * 0.05), _o(self, **kw))[1]
        )
    for _ in range(5):
        orchestrator.cache.clear()
        orchestrator.run(CASE, run_evaluator=False)
    assert len(count) == 5


def test_string_retrieval_queries_are_not_split_into_characters(orchestrator, monkeypatch):
    from src.agents.planner import PlannerAgent

    monkeypatch.setattr(
        PlannerAgent, "run", lambda self, **kw: _result({"retrieval_queries": "online booking"}, "demo-fixture")
    )
    seen = []
    real = orchestrator.retriever.retrieve_many
    monkeypatch.setattr(orchestrator.retriever, "retrieve_many", lambda q, k=None: seen.append(q) or real(q, k))
    orchestrator.run(CASE, run_evaluator=False)
    assert seen == [["online booking"]]


def test_cache_namespace_is_versioned(orchestrator):
    assert orchestrator.cache.dir.name == "pipeline_v2"
