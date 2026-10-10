"""Record the demo fixture from one live Gemini run of the sample bikeshop case.

    python -m scripts.record_demo            # writes data/demo/alexandra_bikeshop.fixture.json

Needs GOOGLE_API_KEY (DEMO_MODE must not be forced on). Six Gemini calls, or
none if the run is already in the pipeline cache. Refuses to write the fixture
if any tier fell back to the mock, or if an agent's output wasn't valid JSON.
"""
from __future__ import annotations

import json
import sys
from datetime import date

from src.orchestrator.pipeline import Orchestrator
from src.utils.config import settings

CASE_PATH = settings.project_root / "data" / "business_cases" / "alexandra_bikeshop.txt"
STAGES = ["planner", "framework", "canvas", "strategist", "roadmap", "evaluator"]
ATTR = {"planner": "plan", "strategist": "strategy", "evaluator": "evaluation"}


def main() -> int:
    """Run the sample case live and write the demo fixture.

    Returns:
        0 on success; 1 if no live key, a tier fell back to mock, or an output was not valid JSON.
    """
    if not settings.google_api_key or settings.demo_active:
        print("Needs GOOGLE_API_KEY and DEMO_MODE not forced on.")
        return 1
    orch = Orchestrator()
    trace = orch.run(CASE_PATH.read_text(encoding="utf-8"))
    if orch.router.fallback_used:
        print("A tier fell back to mock (daily quota); not recording.")
        return 1

    agents = {s: getattr(trace, ATTR.get(s, s)) for s in STAGES}
    for stage, a in agents.items():
        usage = a.extras.get("usage") or {}
        print(f"{stage:<11} {a.model:<24} json={isinstance(a.output, dict)} {usage}")
    bad = [s for s, a in agents.items() if not isinstance(a.output, dict)]
    if bad:
        print(f"Invalid JSON from {bad}; not recording.")
        return 1

    models = sorted({a.model for a in agents.values()})
    fixture = {
        "_note": (
            f"Recorded from a live Gemini run on {date.today().isoformat()}, "
            f"models {', '.join(models)}. Outputs are verbatim; scripts/record_demo.py regenerates it."
        ),
        "case": "alexandra_bikeshop",
        "recorded": {
            "date": date.today().isoformat(),
            "model_per_stage": trace.model_per_stage,
            "timings": trace.timings,
            "usage": {s: a.extras.get("usage") for s, a in agents.items()},
        },
        **{s: a.output for s, a in agents.items()},
    }
    settings.demo_fixture.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {settings.demo_fixture.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
