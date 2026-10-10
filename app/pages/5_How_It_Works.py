"""Plain-language glossary — RAG, embeddings, multi-agent, multi-LLM, optimization."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.edu_content import GLOSSARY  # noqa: E402

st.set_page_config(page_title="How It Works", page_icon="💡", layout="wide")

st.title("💡 How it works")
st.caption("Plain-language explanations of every concept this system uses.")

for entry in GLOSSARY:
    with st.container(border=True):
        st.markdown(f"### {entry['term']}")
        st.write(entry["plain"])
        st.info(f"**In this project:** {entry['why_here']}", icon="🎯")

st.divider()

# ─── Optimization checklist ──────────────────────────────────────────────────
st.subheader("🛠️ Optimisations implemented (FR9)")
st.markdown(
    "These are the explicit cost / latency / robustness measures in the codebase:"
)
st.markdown(
    "1. **Persistent vector DB** — corpus is embedded **once**, saved to `vector_store/index/`, "
    "reloaded in milliseconds on every app start.\n"
    "2. **Prompt mutualization** — every agent's prompt lives in `src/agents/prompts.py` with a "
    "shared header. No ad-hoc prompt rebuilding.\n"
    "3. **Reduced injected context** — top-k passages are capped per stage (the canvas agent only "
    "sees ≤ 4000 chars; the framework agent ≤ 6000). We never send the whole corpus.\n"
    "4. **Pipeline cache** — the full trace is keyed on the business case + evaluator flag and "
    "saved to `.cache/pipeline/`. Re-runs on the same input return instantly.\n"
    "5. **Conditional execution** — the evaluator agent is optional (sidebar toggle).\n"
    "6. **Two-stage architecture** — fast tier for extraction / first-draft synthesis, powerful "
    "tier for strategic reasoning, generation, and evaluation.\n"
    "7. **Quota-aware fallback** — when a tier hits its daily quota (`PerDay` 429), the router "
    "marks it dead and substitutes a Mock client; the pipeline still completes.\n"
    "8. **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs on CPU, zero API cost."
)
