"""RAG Inspector — corpus chunks + interactive retrieval test.

Lets you:
    • see every chunk indexed in FAISS
    • run an arbitrary query and inspect the top-k passages with scores
    • filter by framework
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import Retriever  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.vector_store.faiss_store import FaissStore  # noqa: E402

st.set_page_config(page_title="RAG Inspector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner=False)
def get_store() -> FaissStore:
    s = FaissStore(settings.index_dir)
    s.load()
    return s


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever(store=get_store())


st.title("🔍 RAG Inspector")
st.caption(
    "The vector store is the *memory* the agents draw from. "
    "Here you can inspect every chunk we indexed and test what comes back for any query."
)

store = get_store()
chunks = store.chunks

# ─── Stats ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total chunks", len(chunks))
col2.metric("Embedding dim", store.index.d if store.index else "?")
col3.metric(
    "Avg chunk length (chars)",
    int(sum(len(c.text) for c in chunks) / max(len(chunks), 1)),
)

# ─── Distribution by framework ────────────────────────────────────────────────
st.subheader("📊 Chunks per framework")
counts = Counter(c.framework for c in chunks)
df_counts = pd.DataFrame({"framework": list(counts.keys()), "chunks": list(counts.values())})
st.bar_chart(df_counts.set_index("framework"))

# ─── Live retrieval test ──────────────────────────────────────────────────────
st.divider()
st.subheader("🧪 Try a query")
st.caption("Type any question — see what FAISS returns. This is exactly what the agents see.")

query = st.text_input("Query", value="What are the 7 action fields of the digital transformation canvas?")
col_a, col_b = st.columns([1, 1])
top_k = col_a.slider("Top-k", min_value=1, max_value=10, value=5)
frameworks_in_corpus = sorted({c.framework for c in chunks})
fw_filter = col_b.multiselect("Restrict to frameworks", frameworks_in_corpus, default=[])

if query:
    retriever = get_retriever()
    passages = retriever.retrieve(query, k=top_k, framework_filter=fw_filter or None)
    st.markdown(f"**{len(passages)} passages** retrieved (similarity = cosine over normalized embeddings).")
    for i, p in enumerate(passages, 1):
        with st.container(border=True):
            top = st.columns([3, 1, 1])
            top[0].markdown(f"**{i}. `{p.chunk.framework}` · *{p.chunk.title}* · p.{p.chunk.page}**")
            top[1].metric("Score", f"{p.score:.3f}")
            top[2].metric("Length", f"{len(p.chunk.text)} ch")
            st.write(p.chunk.text)

# ─── Browse all chunks ────────────────────────────────────────────────────────
st.divider()
st.subheader("📑 Browse all indexed chunks")
df = pd.DataFrame(
    [
        {
            "id": c.chunk_id,
            "framework": c.framework,
            "title": c.title,
            "page": c.page,
            "preview": c.text[:160] + "…" if len(c.text) > 160 else c.text,
        }
        for c in chunks
    ]
)
st.dataframe(df, width="stretch", hide_index=True)
