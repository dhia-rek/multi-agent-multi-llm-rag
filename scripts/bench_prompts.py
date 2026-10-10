"""Offline prompt-size comparison, old vs new orchestration (no API calls).

Rebuilds each agent's user prompt from the recorded demo fixture, the way main
did (indent=2 JSON, canvas gets framework insights) and the way the graph does
(compact JSON, canvas without them). Sizes are characters; Gemini averages ~4
chars/token on this text.
"""
from __future__ import annotations

import json

from src.agents import prompts
from src.orchestrator.graph import as_text
from src.utils.config import settings


def old_text(v):
    return json.dumps(v, indent=2, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)


def main() -> None:
    fx = json.loads(settings.demo_fixture.read_text(encoding="utf-8"))
    plan, fw, canvas, strat, road = (fx[k] for k in ("planner", "framework", "canvas", "strategist", "roadmap"))

    rows = {}
    for name, f in (("old", old_text), ("new", as_text)):
        rows[name] = {
            "strategist": len(prompts.STRATEGIST_USER.format(plan=f(plan)[:3000], canvas=f(canvas)[:3000], framework_insights=f(fw)[:3000])),
            "roadmap": len(prompts.ROADMAP_USER.format(strategy=f(strat)[:4000], canvas=f(canvas)[:3000])),
            "evaluator": len(prompts.EVALUATOR_USER.format(business_case="b" * 1500, plan=f(plan)[:2000], strategy=f(strat)[:2000], roadmap=f(road)[:4000])),
        }
    print("chars per prompt (fixture-sized inputs)")
    for k in rows["old"]:
        o, n = rows["old"][k], rows["new"][k]
        print(f"{k:<11} old {o:>6}  new {n:>6}  {100 * (n - o) / o:+.0f}%")


if __name__ == "__main__":
    main()
