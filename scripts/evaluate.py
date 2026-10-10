"""Small evaluation harness over data/eval/cases.jsonl.

Offline (always runs, no API key):
    Index-level:
      - embedder_coverage: share of corpus tokens inside the embedding model's
                           window (tokens past max_seq_length are silently dropped)
      - field_query_p_at_3: for each of the 7 Canvas action fields, ask a
                           field-specific question; share of the top-3 passages
                           that actually discuss that field (lexical check)
    Per case (query = business case text, the pipeline's fallback query):
      - sources:       distinct framework papers among the top-k (max 3)
      - example_share: share of top-k passages taken from the bikeshop example
                       roadmap rather than the framework papers
      - ms:            retrieval latency

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
EXAMPLE_SET = "bikeshop_roadmap"
CANVAS_FIELDS = [
    "Customer Centricity",
    "New Technologies",
    "Cloud & Data",
    "Digital Business Development",
    "Process Engineering",
    "Digital Leadership & Culture",
    "Digital Marketing",
]


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("&", "and").split())


def case_metrics(passages: list[RetrievedPassage]) -> dict:
    return {
        "sources": len({p.chunk.title for p in passages if p.chunk.framework != EXAMPLE_SET}),
        "example_share": (
            sum(p.chunk.framework == EXAMPLE_SET for p in passages) / len(passages) if passages else 0.0
        ),
    }


def embedder_coverage(retriever: Retriever) -> float:
    model = retriever.embedder.model
    window = model.max_seq_length - 2  # [CLS] and [SEP]
    total = visible = 0
    for chunk in retriever.store.chunks:
        n = len(model.tokenizer(chunk.text, add_special_tokens=False)["input_ids"])
        total += n
        visible += min(n, window)
    return visible / total if total else 0.0


def field_query_precision(retriever: Retriever, k: int = 3) -> float:
    hits = 0
    for field in CANVAS_FIELDS:
        passages = retriever.retrieve(f"How should an SME approach {field} in its digital transformation?", k=k)
        hits += sum(_norm(field) in _norm(p.chunk.text) for p in passages)
    return hits / (k * len(CANVAS_FIELDS))


def run_offline(cases: list[dict], k: int) -> dict:
    retriever = Retriever()
    retriever.retrieve("warm-up")  # load the embedding model outside the timings
    rows = []
    for case in cases:
        t = time.perf_counter()
        passages = retriever.retrieve(case["business_case"][:300], k=k)
        ms = (time.perf_counter() - t) * 1000
        rows.append({"id": case["id"], **case_metrics(passages), "ms": ms})
    return {
        "k": k,
        "n_chunks": len(retriever.store.chunks),
        "embedder_coverage": embedder_coverage(retriever),
        "field_query_p_at_3": field_query_precision(retriever),
        "cases": rows,
    }


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
    rows = res["cases"]
    lines = [
        f"Index: {res['n_chunks']} chunks · embedder coverage {res['embedder_coverage']:.0%} "
        f"· field-query P@3 {res['field_query_p_at_3']:.2f}",
        "",
        f"Per-case retrieval, top-{res['k']}, query = business case text",
        "",
        "| case | framework papers in top-k (of 3) | share from example roadmap |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {r['sources']} | {r['example_share']:.2f} |")
    lines.append(f"| **mean** | **{_mean(rows, 'sources'):.1f}** | **{_mean(rows, 'example_share'):.2f}** |")
    lines.append("")
    lines.append(f"Median retrieval latency: {statistics.median(r['ms'] for r in rows):.0f} ms (CPU)")
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
