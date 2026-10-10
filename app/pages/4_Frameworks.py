"""Frameworks reference — what each corpus document brings to the analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.edu_content import FRAMEWORKS  # noqa: E402

st.set_page_config(page_title="Frameworks", page_icon="📚", layout="wide")

st.title("📚 Digital Transformation Frameworks")
st.caption(
    "These are the four documents in `corpus/`. The system uses the first three as the "
    "**knowledge base** (chunked + embedded + indexed) and the fourth as the **business case** input."
)

for key, f in FRAMEWORKS.items():
    with st.container(border=True):
        st.markdown(f"### {f['title']}")
        st.caption(f["source"])
        st.write(f["summary"])
        st.markdown("**Key ideas**")
        for k in f["key_ideas"]:
            st.markdown(f"- {k}")

st.divider()

st.subheader("🧩 How the system uses them together")
st.markdown(
    "- **Wade 2015** answers *Why transform?* — the Planner agent extracts the drivers "
    "(customer pressure, competition, technology…).\n"
    "- **Peter 2018** (7 action fields) answers *What to transform?* — the Canvas Analysis "
    "agent grades maturity field by field.\n"
    "- **Elia 2024** (Strategy / Pillars / Value / Pitfalls) answers *How to integrate strategy* — "
    "the Strategist agent uses it to articulate purpose, pillars, value, risks.\n"
    "- **Peter 2024** (Bikeshop) is treated as an **input case** — a realistic SME stand-in "
    "you can test the pipeline against.\n"
)
