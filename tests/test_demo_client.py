import json
import re
from pathlib import Path

import pytest

from src.agents import prompts
from src.llm_clients.demo import DemoLLMClient
from src.utils.config import settings

SYSTEMS = [
    prompts.PLANNER_SYSTEM,
    prompts.FRAMEWORK_SYSTEM,
    prompts.CANVAS_SYSTEM,
    prompts.STRATEGIST_SYSTEM,
    prompts.ROADMAP_SYSTEM,
    prompts.EVALUATOR_SYSTEM,
]


@pytest.fixture(scope="module")
def fixture_data():
    return json.loads(Path(settings.demo_fixture).read_text(encoding="utf-8"))


@pytest.mark.parametrize("system", SYSTEMS)
def test_returns_valid_json_for_each_agent(system):
    resp = DemoLLMClient(tier="fast").generate("ignored", system=system)
    assert isinstance(json.loads(resp.text), dict)
    assert resp.model == "demo-fixture-fast"
    assert resp.tier == "fast"


@pytest.mark.parametrize("system", ["something else", "", None])
def test_unknown_system_prompt_raises_keyerror(system):
    with pytest.raises(KeyError):
        DemoLLMClient().generate("x", system=system)


def test_canvas_schema(fixture_data):
    fields = fixture_data["canvas"]["fields"]
    assert len(fields) == 7
    for f in fields:
        assert isinstance(f["maturity_score"], int) and 1 <= f["maturity_score"] <= 5
        assert {"name", "current_state", "gaps", "opportunities"} <= f.keys()


def test_evaluator_schema(fixture_data):
    score = fixture_data["evaluator"]["overall_score"]
    assert isinstance(score, int) and 1 <= score <= 10


def _template_keys(block: str) -> set[str]:
    """Keys quoted in a JSON-ish snippet of the prompt (`"key":`)."""
    return set(re.findall(r'"(\w+)":', block))


def test_roadmap_schema_matches_prompt(fixture_data):
    template = prompts.ROADMAP_USER
    phase_keys = {"name", "objectives", "initiatives"}
    initiative_keys = {
        "name", "owner", "kpis", "milestones", "duration_weeks",
        "budget_estimate", "framework_anchor", "dependencies",
    }  # fmt: skip
    # guard: the expected keys really are the ones the prompt asks for
    assert (phase_keys | initiative_keys | {"horizon_months", "phases", "kpi_summary"}) <= _template_keys(
        template
    )

    roadmap = fixture_data["roadmap"]
    assert {"horizon_months", "phases", "kpi_summary"} <= roadmap.keys()
    assert roadmap["phases"]
    for phase in roadmap["phases"]:
        assert phase_keys <= phase.keys()
        assert phase["initiatives"]
        for ini in phase["initiatives"]:
            assert initiative_keys <= ini.keys()
