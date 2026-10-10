"""Home page — architecture overview, status, and what each layer does.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src.*` and `app.*` importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.edu_content import AGENTS, PIPELINE_BLOCKS  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.vector_store.faiss_store import FaissStore  # noqa: E402

st.set_page_config(
    page_title="DT Roadmap — Multi-Agent RAG (Edu)",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def index_info() -> dict:
    return FaissStore(settings.index_dir).info()


# ─── Sidebar: system status ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🧭 DT Roadmap")
    st.caption("Multi-Agent · Multi-LLM · Local RAG")

    st.subheader("System status")
    info = index_info()
    if info:
        st.success(f"Vector index: **{info.get('n_chunks', 0)}** chunks · dim {info.get('dim', '?')}")
        st.caption("Frameworks indexed: " + ", ".join(info.get("frameworks", [])))
    else:
        st.error("Vector index missing — run `python -m scripts.build_index`")

    if settings.google_api_key:
        st.success("Gemini API key detected")
    else:
        st.warning("No GOOGLE_API_KEY — running in MOCK mode")

    st.markdown(f"**Fast tier:** `{settings.gemini_fast_model}`")
    st.markdown(f"**Powerful tier:** `{settings.gemini_powerful_model}`")
    if settings.ollama_model:
        st.markdown(f"**Local tier:** `{settings.ollama_model}` (Ollama)")
    else:
        st.caption("Local tier: not configured")

    st.divider()
    st.caption("Use the **pages** in the sidebar to explore.")


# ─── Main: Hero + architecture ────────────────────────────────────────────────
st.title("🧭 Multi-Agent · Multi-LLM RAG System")
st.markdown(
    "**Goal:** turn a free-text business case into a *credible, framework-grounded* "
    "digital transformation roadmap, by chaining six specialised agents over a local RAG corpus, "
    "with explicit Multi-LLM routing and optimisation."
)

st.subheader("📐 Architecture at a glance")
cols = st.columns(5)
for col, block in zip(cols, PIPELINE_BLOCKS):
    with col:
        st.markdown(f"### {block['icon']}")
        st.markdown(f"**{block['name']}**")
        st.caption(block["what"])

with st.expander("📖 Why each layer exists", expanded=False):
    for block in PIPELINE_BLOCKS:
        st.markdown(f"**{block['icon']} {block['name']}** — {block['why']}")

st.divider()

# ─── Stage diagram (mermaid) ──────────────────────────────────────────────────
st.subheader("🔁 The pipeline")
st.markdown(
    "Here's exactly what runs every time you click **Run pipeline**. "
    "Each stage produces a structured JSON object that the next stage reads."
)
st.graphviz_chart(
    """
    digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
      bc [label="Business case\\n(free text)", fillcolor="#fff3cd"];
      pl [label="1. Planner\\n(fast)"];
      rg [label="2. Retrieval\\n(FAISS)", fillcolor="#d4edda"];
      fw [label="3. Framework\\n(fast)"];
      cv [label="4. Canvas Analysis\\n(powerful)"];
      st [label="5. Strategist\\n(powerful)"];
      rm [label="6. Roadmap\\n(powerful)"];
      ev [label="7. Evaluator\\n(powerful, optional)", fillcolor="#fde2e2"];
      out [label="Structured roadmap\\n(JSON)", fillcolor="#fff3cd"];
      bc -> pl -> rg -> fw -> cv -> st -> rm -> ev -> out;
    }
    """,
    use_container_width=True,
)

st.divider()

# ─── Agent cards ──────────────────────────────────────────────────────────────
st.subheader("🤖 Meet the six agents")
st.caption("Each has one role, one prompt, one expected output. Click for details.")
for key, agent in AGENTS.items():
    with st.expander(f"{agent['icon']}  **{agent['name']}** — {agent['purpose']}"):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**What it does**")
            st.write(agent["what_it_does"])
            st.markdown(f"**Anchored in:** {agent['framework_anchor']}")
        with c2:
            st.markdown("**Routing profile**")
            st.markdown(
                f"- task type: `{agent['task_type']}`\n"
                f"- complexity: `{agent['complexity']}`\n"
                f"- criticality: `{agent['criticality']}`\n"
                f"- → routes to **{agent['expected_tier']}** tier"
            )
            st.caption(agent["tier_rationale"])

st.divider()

st.info(
    "👉 Open **🚀 Run Pipeline** in the sidebar to actually run the system on a business case "
    "and watch each stage execute live with explanations.",
    icon="ℹ️",
)
