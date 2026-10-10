import json

STAGES = ["planner", "framework", "canvas", "strategist", "roadmap", "evaluator"]
CASE = "Alexandra runs a small bike shop with two stores and wants to digitize repairs."


def test_full_run_in_demo_mode(orchestrator):
    trace = orchestrator.run(CASE)

    assert set(trace.model_per_stage) == set(STAGES)
    assert all(m.startswith("demo-fixture") for m in trace.model_per_stage.values())
    for stage in (*STAGES, "retrieval"):
        assert stage in trace.timings
    assert trace.retrieval  # fake index returned passages
    assert isinstance(trace.roadmap.output, dict)
    assert trace.roadmap.output["phases"]
    assert trace.evaluation is not None

    json.dumps(trace.as_dict())  # must be serialisable


def test_streaming_yields_every_stage_in_order(orchestrator):
    stages = [s for s, _ in orchestrator.run_streaming(CASE)]
    assert stages == ["planner", "retrieval", "framework", "canvas", "strategist", "roadmap", "evaluator", "done"]


def test_without_evaluator(orchestrator):
    trace = orchestrator.run(CASE, run_evaluator=False)
    assert trace.evaluation is None
    assert "evaluator" not in trace.model_per_stage


def test_second_run_hits_cache(orchestrator):
    first = orchestrator.run(CASE)
    stages = [s for s, _ in orchestrator.run_streaming(CASE)]
    assert stages == ["cached", "done"]
    assert orchestrator.run(CASE).as_dict() == first.as_dict()


def test_evaluator_flag_is_part_of_cache_key(orchestrator):
    orchestrator.run(CASE, run_evaluator=False)
    stages = [s for s, _ in orchestrator.run_streaming(CASE, run_evaluator=True)]
    assert stages[0] == "planner"
