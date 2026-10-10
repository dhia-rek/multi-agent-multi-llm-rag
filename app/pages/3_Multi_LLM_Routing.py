"""Multi-LLM Routing page — explain and test the routing layer."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.edu_content import AGENTS  # noqa: E402
from src.routing.router import Router  # noqa: E402
from src.utils.config import settings  # noqa: E402

st.set_page_config(page_title="Multi-LLM Routing", page_icon="🧭", layout="wide")

st.title("🧭 Multi-LLM Routing")
st.caption(
    "The router is the **decision layer** that picks which model to call for each task. "
    "Goal: use a cheap model where it's enough, a powerful one where it matters."
)

# ─── Tier overview ────────────────────────────────────────────────────────────
st.subheader("Tiers")
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("### ⚡ Fast")
    st.code(settings.gemini_fast_model, language="text")
    st.caption("Used for extraction, reformulation, first-draft synthesis. Highest free-tier quota.")
with t2:
    st.markdown("### 🧠 Powerful")
    st.code(settings.gemini_powerful_model, language="text")
    st.caption("Used for strategic reasoning, final generation, evaluation.")
with t3:
    st.markdown("### 🏠 Local")
    if settings.ollama_model:
        st.code(settings.ollama_model, language="text")
        st.caption("Optional Ollama model — repetitive / low-criticality tasks.")
    else:
        st.code("(not configured)", language="text")
        st.caption("Set `OLLAMA_MODEL` in `.env` to enable.")

st.divider()

# ─── Routing policy ───────────────────────────────────────────────────────────
st.subheader("🎯 Routing policy")
st.markdown(
    "The router decides based on three explicit attributes each agent declares about itself:"
)
st.markdown(
    "- **`task_type`** — extraction · synthesis · reasoning · generation · evaluation\n"
    "- **`complexity`** — low · medium · high\n"
    "- **`criticality`** — low · medium · high\n"
)
st.markdown(
    "Rules (in `src/routing/router.py`):\n\n"
    "1. If `task_type ∈ {reasoning, generation, evaluation}` *or* `complexity = high` *or* `criticality = high` → **powerful**\n"
    "2. Else if `criticality = low` *and* a local model is configured → **local**\n"
    "3. Otherwise → **fast**"
)

st.divider()

# ─── Per-agent table ──────────────────────────────────────────────────────────
st.subheader("📋 What each agent gets routed to")
rows = []
for _key, a in AGENTS.items():
    rows.append(
        {
            "agent": f"{a['icon']} {a['name']}",
            "task_type": a["task_type"],
            "complexity": a["complexity"],
            "criticality": a["criticality"],
            "→ tier": a["expected_tier"],
            "model (current)": (
                settings.gemini_powerful_model if a["expected_tier"] == "powerful"
                else settings.gemini_fast_model if a["expected_tier"] == "fast"
                else (settings.ollama_model or "(local n/a)")
            ),
        }
    )
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ─── Live routing tester ──────────────────────────────────────────────────────
st.subheader("🧪 Try the router live")
st.caption("Pick attributes — see which tier you'd land in.")
col1, col2, col3 = st.columns(3)
task_type = col1.selectbox(
    "task_type",
    ["extraction", "synthesis", "reasoning", "generation", "evaluation"],
)
complexity = col2.selectbox("complexity", ["low", "medium", "high"], index=1)
criticality = col3.selectbox("criticality", ["low", "medium", "high"], index=1)

router = Router()
decision = router.decide(task_type, complexity, criticality)

st.success(f"→ Routed to **{decision.tier}** tier")
st.caption(decision.reason)
selected_model = {
    "fast": settings.gemini_fast_model,
    "powerful": settings.gemini_powerful_model,
    "local": settings.ollama_model or "(local not configured)",
}[decision.tier]
st.markdown(f"**Model that would be called:** `{selected_model}`")

st.divider()

# ─── Why this matters ─────────────────────────────────────────────────────────
st.subheader("💡 Why bother with routing?")
st.markdown(
    "- **Cost** — extraction tasks on Flash are ~10–20× cheaper than on Pro.\n"
    "- **Latency** — Flash answers in 1–2s; Pro can take 5–10s on long contexts.\n"
    "- **Quota** — free tiers are tight on Pro (~50/day). Routing lets the powerful model "
    "do only the strategic work.\n"
    "- **Robustness** — when a tier exhausts its daily quota, the router falls back to a mock "
    "client so the pipeline still completes (see `mark_exhausted` in `router.py`)."
)
