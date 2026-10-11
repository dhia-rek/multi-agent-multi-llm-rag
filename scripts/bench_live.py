"""Live latency/token benchmark of one full pipeline run on the sample bikeshop case.

    python -m scripts.bench_live <label>      # needs GOOGLE_API_KEY; writes .agent-scratch/bench-<label>.json

Bypasses the pipeline cache. About 6 Gemini calls (more if a stage is retried). Run it on `main`
and on this branch to get the before/after numbers in the README.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.orchestrator.pipeline import Orchestrator

CASE_PATH = Path("data/business_cases/alexandra_bikeshop.txt")
TOKEN_FIELDS = ("prompt_tokens", "output_tokens", "thinking_tokens")


def main(label: str) -> dict[str, Any]:
    """Run the pipeline once without cache and write the numbers to `.agent-scratch/bench-<label>.json`.

    Args:
        label: Free-text tag for the run (e.g. the branch name); used in the output file name.

    Returns:
        The result dict that was written to disk.
    """
    orch = Orchestrator()
    # Replace the cache methods on the instance so every stage hits the live model.
    cache: Any = orch.cache
    cache.get = lambda k: None
    cache.set = lambda k, v: None
    start = time.time()
    trace = orch.run(CASE_PATH.read_text())
    wall = round(time.time() - start, 2)

    stages = {
        "planner": trace.plan,
        "framework": trace.framework,
        "canvas": trace.canvas,
        "strategist": trace.strategy,
        "roadmap": trace.roadmap,
        "evaluator": trace.evaluation,
    }
    usage = {name: (r.extras.get("usage") or {}) if r else {} for name, r in stages.items()}
    totals = {f: sum((u.get(f) or 0) for u in usage.values()) for f in TOKEN_FIELDS}
    roadmap = trace.roadmap.output if trace.roadmap else None
    evaluation = trace.evaluation.output if trace.evaluation else None
    result = {
        "label": label,
        "wall_s": wall,
        "timings": trace.timings,
        "usage": usage,
        "total": totals,
        "json_ok": {name: r is not None and isinstance(r.output, (dict, list)) for name, r in stages.items()},
        "fallback": orch.router.fallback_used,
        "roadmap_phases": len(roadmap.get("phases", [])) if isinstance(roadmap, dict) else None,
        "eval_score": evaluation.get("overall_score") if isinstance(evaluation, dict) else None,
    }
    Path(".agent-scratch").mkdir(exist_ok=True)
    Path(f".agent-scratch/bench-{label}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.bench_live <label>")
    print(json.dumps(main(sys.argv[1])))
