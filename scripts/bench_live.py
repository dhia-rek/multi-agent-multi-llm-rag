"""Live latency/token benchmark of one full pipeline run on the sample bikeshop case.

    python -m scripts.bench_live <label>      # needs GOOGLE_API_KEY; writes .agent-scratch/bench-<label>.json

Bypasses the pipeline cache. About 6 Gemini calls (more if a stage is retried). Run it on `main`
and on this branch to get the before/after numbers in the README.
"""
import json
import sys
import time
from pathlib import Path

from src.orchestrator.pipeline import Orchestrator

LABEL = sys.argv[1]
case = Path("data/business_cases/alexandra_bikeshop.txt").read_text()
o = Orchestrator()
o.cache.get = lambda k: None  # bypass cache
o.cache.set = lambda k, v: None
t = time.time()
tr = o.run(case)
wall = round(time.time() - t, 2)
ag = {"planner": tr.plan, "framework": tr.framework, "canvas": tr.canvas, "strategist": tr.strategy, "roadmap": tr.roadmap, "evaluator": tr.evaluation}
usage = {k: (a.extras.get("usage") or {}) for k, a in ag.items()}
tot = {f: sum((u.get(f) or 0) for u in usage.values()) for f in ("prompt_tokens", "output_tokens", "thinking_tokens")}
res = {"label": LABEL, "wall_s": wall, "timings": tr.timings, "usage": usage, "total": tot,
       "json_ok": {k: isinstance(a.output, (dict, list)) for k, a in ag.items()},
       "fallback": o.router.fallback_used, "roadmap_phases": len(tr.roadmap.output.get("phases", [])) if isinstance(tr.roadmap.output, dict) else None,
       "eval_score": tr.evaluation.output.get("overall_score") if isinstance(tr.evaluation.output, dict) else None}
Path(".agent-scratch").mkdir(exist_ok=True)
Path(f".agent-scratch/bench-{LABEL}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res))
