"""Small evaluation harness over data/eval/cases.jsonl.

Offline (always runs, no API key):
    Retrieval quality per case, using the business case itself as the query
    (the pipeline's fallback when the planner gives no queries):
      - field_recall:  share of the case's expected Canvas action fields whose
                       name appears in the context the agents receive (top-k
                       passages capped at 6000 chars, as in the pipeline;
                       lexical proxy)
      - sources:       distinct framework papers among the top-k (max 3)
      - example_share: share of top-k passages taken from the bikeshop example
                       roadmap rather than the framework papers
      - ms:            retrieval latency
    Reported twice: whole corpus, and framework papers only (example excluded).

Live (only with GOOGLE_API_KEY and --live):
    Full pipeline per case: JSON parse rate across the six agents, evaluator
    score, canvas priority fields vs expected fields, end-to-end latency.

Usage:
    python -m scripts.evaluate            # offline retrieval table
    python -m scripts.evaluate --live     # + full pipeline (needs a key)
    python -m scripts.evaluate --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.retrieval.retriever import RetrievedPassage, Retriever
from src.utils.config import settings

CASES_PATH = settings.project_root / "data" / "eval" / "cases.jsonl"
FRAMEWORK_PAPERS = ["dt_framework"]
EXAMPLE_SET = "bikeshop_roadmap"
CONTEXT_CHARS = 6000


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("&", "and").split())


def retrieval_metrics(passages: list[RetrievedPassage], expected_fields: list[str]) -> dict:
    # Judge only what the agents actually see: the pipeline caps context at 6000 chars.
    blob = _norm(Retriever.format_context(passages)[:CONTEXT_CHARS])
    hits = [f for f in expected_fields if _norm(f) in blob]
    return {
        "field_recall": len(hits) / len(expected_fields) if expected_fields else 0.0,
        "sources": len({p.chunk.title for p in passages if p.chunk.framework != EXAMPLE_SET}),
        "example_share": (
            sum(p.chunk.framework == EXAMPLE_SET for p in passages) / len(passages) if passages else 0.0
        ),
    }


def run_offline(cases: list[dict], k: int) -> dict:
    retriever = Retriever()
    retriever.retrieve("warm-up")  # load the embedding model outside the timings
    results: dict[str, list[dict]] = {"all": [], "papers_only": []}
    for case in cases:
        query = case["business_case"][:300]
        for setting, filt in (("all", None), ("papers_only", FRAMEWORK_PAPERS)):
            t = time.perf_counter()
            passages = retriever.retrieve(query, k=k, framework_filter=filt)
            ms = (time.perf_counter() - t) * 1000
            row = {"id": case["id"], **retrieval_metrics(passages, case["expected_fields"]), "ms": ms}
            results[setting].append(row)
    return {"k": k, "n_chunks": len(retriever.store.chunks), **results}


def run_live(cases: list[dict]) -> list[dict]:
    from src.orchestrator.pipeline import Orchestrator

    orch = Orchestrator()
    rows = []
    for case in cases:
        t = time.perf_counter()
        trace = orch.run(case["business_case"])
        secs = time.perf_counter() - t
        agents = [trace.plan, trace.framework, trace.canvas, trace.strategy, trace.roadmap, trace.evaluation]
        parsed = sum(isinstance(a.output, dict) for a in agents if a)
        canvas = trace.canvas.output if isinstance(trace.canvas.output, dict) else {}
        priority = {_norm(f) for f in canvas.get("priority_fields", [])}
        expected = {_norm(f) for f in case["expected_fields"]}
        evaluation = trace.evaluation.output if isinstance(trace.evaluation.output, dict) else {}
        rows.append(
            {
                "id": case["id"],
                "json_ok": f"{parsed}/{len(agents)}",
                "priority_hit": len(priority & expected) / len(priority) if priority else 0.0,
                "eval_score": evaluation.get("overall_score"),
                "seconds": round(secs, 1),
            }
        )
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows)


def offline_table(res: dict) -> str:
    lines = [
        f"Retrieval, top-{res['k']} over {res['n_chunks']} chunks, query = business case text",
        "",
        "| case | field recall | papers in top-k | example share | field recall (papers only) |",
        "|---|---|---|---|---|",
    ]
    for a, p in zip(res["all"], res["papers_only"], strict=True):
        lines.append(
            f"| {a['id']} | {a['field_recall']:.2f} | {a['sources']} | {a['example_share']:.2f} "
            f"| {p['field_recall']:.2f} |"
        )
    a, p = res["all"], res["papers_only"]
    lines.append(
        f"| **mean** | **{_mean(a, 'field_recall'):.2f}** | **{_mean(a, 'sources'):.1f}** "
        f"| **{_mean(a, 'example_share'):.2f}** | **{_mean(p, 'field_recall'):.2f}** |"
    )
    lines.append("")
    lines.append(f"Median retrieval latency: {statistics.median(r['ms'] for r in a):.0f} ms (CPU)")
    return "\n".join(lines)


def live_table(rows: list[dict]) -> str:
    lines = [
        "| case | agents with valid JSON | priority fields hit | evaluator score | seconds |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['json_ok']} | {r['priority_hit']:.2f} | {r['eval_score']} | {r['seconds']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--live", action="store_true", help="also run the full pipeline (needs GOOGLE_API_KEY)")
    ap.add_argument("--json", type=Path, help="write raw results to this file")
    args = ap.parse_args()

    cases = load_cases()
    out: dict = {"offline": run_offline(cases, args.k)}
    print(offline_table(out["offline"]))

    if args.live:
        if not settings.google_api_key or settings.demo_active:
            print("\nLive evaluation skipped: needs GOOGLE_API_KEY and DEMO_MODE not forced on.")
        else:
            out["live"] = run_live(cases)
            print("\n" + live_table(out["live"]))

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
