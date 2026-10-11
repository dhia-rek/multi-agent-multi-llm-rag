# Running and configuration

## Install with pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned export of uv.lock
python -m scripts.build_index
streamlit run app/streamlit_app.py
```

## REST API

```bash
uv run uvicorn api.main:app --port 8000   # Swagger UI at http://localhost:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, configured models, index ready, `demo_mode` |
| GET | `/pipeline-info` | Models per tier, embedding model, chunking, top-k |
| POST | `/analyze` | Planner → RAG → Framework and Canvas in parallel |
| POST | `/generate-roadmap` | Full pipeline: roadmap, optional evaluation, trace |

```bash
curl -s -X POST localhost:8000/generate-roadmap -H 'Content-Type: application/json' \
  -d '{"business_case": "Family bakery, 12 staff, wants online pre-orders and a loyalty app.", "run_evaluator": false}'
```

## Offline mode (no API key)

| `DEMO_MODE` | Behaviour |
|---|---|
| `auto` (default) | Offline mode when `GOOGLE_API_KEY` is empty |
| `on` | Always offline mode, even with a key |
| `off` | Never: without a key, agents return placeholder mock text |

In offline mode, retrieval runs for real, while the six agents replay [`data/demo/alexandra_bikeshop.fixture.json`](../data/demo/alexandra_bikeshop.fixture.json). **That fixture was recorded from a live Gemini run on 2026-10-10** (`gemini-2.5-flash-lite` for planner and framework, `gemini-2.5-flash` for the other four agents). Outputs are verbatim, with per-stage timings and token usage in its `recorded` block. Regenerate it with `python -m scripts.record_demo` (6 Gemini calls). Any other business case will still show bikeshop outputs, and the UI says so. Offline traces are cached separately from live ones.

## LangSmith tracing (optional)

Tracing is off unless you set the standard LangChain variables, for example in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU region
LANGSMITH_PROJECT=multi-agent-rag
```

Each pipeline run becomes one `LangGraph` trace with a span per agent node, the Gemini calls, prompts and token usage; the parallel framework/canvas branches show side by side. Works in offline mode too (the fixture client is traced as a chat model). Nothing else to configure.

## Configuration

| Variable | Default | |
|---|---|---|
| `GOOGLE_API_KEY` | – | Enables Gemini |
| `GEMINI_FAST_MODEL` | `gemini-2.5-flash-lite` | Fast tier |
| `GEMINI_POWERFUL_MODEL` | `gemini-2.5-flash` | Powerful tier |
| `GEMINI_THINKING_BUDGET` | `1024` | Thinking-token cap for the powerful tier, added on top of the answer budget |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | – / `http://localhost:11434` | Optional local tier |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `RAG_TOP_K` | `900` / `150` / `6` | Retrieval |
| `DEMO_MODE` | `auto` | See above |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT` | off | Optional tracing, see above |
