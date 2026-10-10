"""Live pipeline runner with educational annotations.

Each stage is rendered as it completes, alongside:
    • a plain-language explanation of what just happened
    • the routing decision (which tier was picked, and why)
    • the model that actually answered
    • the agent's parsed output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.edu_content import AGENTS  # noqa: E402
from src.orchestrator.pipeline import Orchestrator, PipelineTrace  # noqa: E402
from src.utils.config import settings  # noqa: E402

st.set_page_config(page_title="Run pipeline", page_icon="🚀", layout="wide")


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


# ─── Helpers ──────────────────────────────────────────────────────────────────
STAGE_TO_AGENT_KEY = {
    "planner": "planner",
    "framework": "framework",
    "canvas": "canvas_analysis",
    "strategist": "strategist",
    "roadmap": "roadmap",
    "evaluator": "evaluator",
}


def _agent_card(stage: str, agent_result, *, show_raw: bool = False):
    """Render a finished agent stage as a teachable block."""
    meta = AGENTS[STAGE_TO_AGENT_KEY[stage]]
    with st.container(border=True):
        st.markdown(f"### {meta['icon']} {meta['name']}")
        st.caption(meta["purpose"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Routing tier", agent_result.tier)
        c2.metric("Model used", agent_result.model)
        c3.metric("Task type", meta["task_type"])
        st.caption(f"🧭 **Why this tier?** {agent_result.routing.reason}")

        st.markdown("**📤 Output**")
        if isinstance(agent_result.output, (dict, list)):
            st.json(agent_result.output, expanded=False)
        else:
            st.write(agent_result.output)

        if show_raw:
            with st.expander("Raw model text"):
                st.code(agent_result.raw_text, language="text")


def _retrieval_card(trace: PipelineTrace):
    with st.container(border=True):
        st.markdown("### 📚 RAG Retrieval")
        st.caption(
            "FAISS searches the indexed corpus for chunks most similar (cosine) to the planner's "
            "queries. Higher score = closer match."
        )
        st.metric("Passages retrieved", len(trace.retrieval))
        for i, p in enumerate(trace.retrieval[:6], 1):
            with st.expander(
                f"{i}. `{p.chunk.framework}` · *{p.chunk.title}* · p.{p.chunk.page} · score {p.score:.3f}"
            ):
                st.write(p.chunk.text[:600] + "…")


def _roadmap_card(roadmap_dict):
    if not isinstance(roadmap_dict, dict):
        st.write(roadmap_dict)
        return
    st.subheader("🗺️ Final Roadmap")
    if "horizon_months" in roadmap_dict:
        st.caption(f"Horizon: {roadmap_dict.get('horizon_months')} months")
    for phase in roadmap_dict.get("phases", []):
        st.markdown(f"### {phase.get('name', 'Phase')}")
        if phase.get("objectives"):
            st.markdown("**Objectives**")
            for o in phase["objectives"]:
                st.markdown(f"- {o}")
        for init in phase.get("initiatives", []):
            with st.container(border=True):
                st.markdown(f"**{init.get('name', 'Initiative')}**")
                cols = st.columns(4)
                cols[0].metric("Owner", init.get("owner", "—"))
                cols[1].metric("Duration", f"{init.get('duration_weeks', '—')} w")
                cols[2].metric("Budget", str(init.get("budget_estimate", "—")))
                cols[3].metric("Anchor", str(init.get("framework_anchor", "—")))
                if init.get("kpis"):
                    st.markdown("**KPIs:** " + ", ".join(init["kpis"]))
                if init.get("milestones"):
                    st.markdown("**Milestones:** " + " → ".join(init["milestones"]))
                if init.get("dependencies"):
                    st.caption("Depends on: " + ", ".join(init["dependencies"]))
    if roadmap_dict.get("kpi_summary"):
        st.markdown("**Program KPIs:** " + " · ".join(roadmap_dict["kpi_summary"]))


# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Run options")
    run_evaluator = st.checkbox(
        "Run evaluator (optional)",
        value=True,
        help="The evaluator agent is conditional. Turn it off to save 1 LLM call per run (FR9 optimization).",
    )
    show_raw = st.checkbox("Show raw model text", value=False)
    if st.button("🗑 Clear pipeline cache"):
        get_orchestrator().cache.clear()
        st.toast("Cache cleared.")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🚀 Run the pipeline (live)")
st.caption(
    "Submit a business case below. Each stage will appear as it finishes, "
    "with the model used, the routing reason, and the structured output."
)

example_path = ROOT / "data" / "business_cases" / "alexandra_bikeshop.txt"
default_text = (
    example_path.read_text() if example_path.exists()
    else "Our company wants to improve customer experience and reduce operational costs."
)

if settings.demo_active:
    st.warning(
        "**Demo mode** — no Gemini key is configured, so the six agents answer from a "
        "hand-written fixture for the sample bikeshop case (retrieval still runs for real). "
        "Other business cases will still show the bikeshop outputs. "
        "Set `GOOGLE_API_KEY` to run live models.",
        icon="🎭",
    )

business_case = st.text_area("📥 Business case", value=default_text, height=220)
launch = st.button("▶ Run pipeline", type="primary")

if launch and business_case.strip():
    orch = get_orchestrator()
    progress = st.progress(0, text="Starting…")
    total_stages = 7 if run_evaluator else 6
    rendered = set()

    placeholder = st.container()

    for i, (stage, trace) in enumerate(
        orch.run_streaming(business_case, run_evaluator=run_evaluator), start=1
    ):
        progress.progress(min(i / total_stages, 1.0), text=f"Stage: {stage}")

        with placeholder:
            if stage == "cached":
                st.info(
                    "💾 Cache HIT — this exact business case ran before, returning cached trace. "
                    "Click **Clear pipeline cache** in the sidebar to force a fresh run.",
                    icon="ℹ️",
                )
                continue
            if stage == "planner" and "planner" not in rendered:
                _agent_card("planner", trace.plan, show_raw=show_raw)
                rendered.add("planner")
            elif stage == "retrieval":
                _retrieval_card(trace)
            elif stage == "framework":
                _agent_card("framework", trace.framework, show_raw=show_raw)
            elif stage == "canvas":
                _agent_card("canvas", trace.canvas, show_raw=show_raw)
            elif stage == "strategist":
                _agent_card("strategist", trace.strategy, show_raw=show_raw)
            elif stage == "roadmap":
                _agent_card("roadmap", trace.roadmap, show_raw=show_raw)
                if isinstance(trace.roadmap.output, dict):
                    _roadmap_card(trace.roadmap.output)
            elif stage == "evaluator":
                _agent_card("evaluator", trace.evaluation, show_raw=show_raw)
            elif stage == "done":
                progress.progress(1.0, text="Done")
                st.success(
                    f"✅ Finished in {round(sum(trace.timings.values()), 1)}s "
                    f"across {len(trace.timings)} stages."
                )

                # ─── Cost & efficiency summary (FR8 + FR9 visualisation) ──
                st.divider()
                st.subheader("📊 Cost & efficiency summary")
                st.caption(
                    "Snapshot of which tier did what and how long it took. "
                    "Lets you see the multi-LLM routing and the optimisations in action."
                )
                import pandas as pd  # local import keeps the page boot cheap

                results = {
                    "planner": trace.plan,
                    "framework": trace.framework,
                    "canvas": trace.canvas,
                    "strategist": trace.strategy,
                    "roadmap": trace.roadmap,
                    "evaluator": trace.evaluation,
                }
                rows = []
                for stage_name, secs in trace.timings.items():
                    model = trace.model_per_stage.get(stage_name, "—")
                    result = results.get(stage_name)
                    tier = "rag" if stage_name == "retrieval" else (result.tier if result else "—")
                    rows.append(
                        {
                            "stage": stage_name,
                            "model": model,
                            "tier": tier,
                            "seconds": secs,
                        }
                    )
                df = pd.DataFrame(rows)
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.dataframe(df, hide_index=True, width="stretch")
                with c2:
                    st.bar_chart(df.set_index("stage")["seconds"])

                # Download the full trace
                st.download_button(
                    "💾 Download full trace (JSON)",
                    data=json.dumps(trace.as_dict(), indent=2, ensure_ascii=False),
                    file_name="pipeline_trace.json",
                    mime="application/json",
                )
elif not launch:
    st.info("Edit the case above and press **Run pipeline** to start.", icon="👆")
