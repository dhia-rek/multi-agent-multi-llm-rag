# Digital Transformation Roadmap Generator

Six LLM agents and a local RAG pipeline turn a plain-text business case into a structured digital transformation roadmap, grounded in three academic frameworks. Each agent is routed to the cheapest model that can do its job.

[![CI](https://github.com/dhia-rek/multi-agent-multi-llm-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/dhia-rek/multi-agent-multi-llm-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![The six agents running on the sample bike shop case](docs/img/demo.gif)

<sub>Recorded in offline mode: retrieval runs for real and the agent outputs are replayed from a recorded Gemini run.</sub>

## What it does

You describe a company in plain text, for example *"a Swiss bike shop with 8 staff, a basic Shopify site, phone-only repair booking and a CHF 60k budget"*. The system then:

1. **Plans:** extracts stakes, objectives and constraints, and writes retrieval queries.
2. **Retrieves:** searches a local FAISS index of three digital transformation frameworks (Wade 2015, Peter's Digital Transformation Canvas, Elia et al. 2024).
3. **Reasons in stages:** four agents classify the evidence, score maturity on the 7 Canvas action fields, build a strategy and produce a phased roadmap.
4. **Reviews:** an optional evaluator agent scores the roadmap against the frameworks.
5. **Routes:** each agent declares its task type, complexity and criticality, and a router sends it to a fast, powerful or local model.

The output is strict JSON (phases, initiatives, owners, KPIs, milestones, budget and framework references) plus a trace of which model ran each stage, why, and on which passages.

## Results

| | Result |
|---|---|
| Valid JSON from every agent | 18 / 18 on 3 live cases |
| Evaluator score | 6.0 / 10 on average |
| End-to-end time | 45 s on average (free Gemini tier) |
| Corpus actually embedded, after fixing the chunker | 36% → 99% |
| Retrieval precision@3 on the 7 Canvas fields | 0.19 → 0.48 |
| Retrieval latency, 4 queries, batched (CPU) | 29.9 ms → 11.1 ms |

How these were measured, and what they don't show: [docs/evaluation.md](docs/evaluation.md).

## How it works

- **Orchestration:** a LangGraph `StateGraph` with one node per agent and typed JSON between them. The framework and canvas agents run in parallel; the rest is a chain.
- **Multi-LLM routing:** `src/routing/router.py` maps each agent to Gemini Flash-Lite, Gemini Flash or a local Ollama model. Rate limits are retried; if a tier runs out of daily quota, it falls back so the run still finishes.
- **Local RAG:** pypdf, paragraph-aware chunking, MiniLM embeddings on CPU and a FAISS cosine index. Retrieval costs nothing per query.
- **Offline mode:** without an API key, retrieval runs for real and the agents replay a recorded Gemini run, so the app, the API and CI all work without keys.

Diagrams, the agent table, the routing policy and the old-vs-new comparison: [docs/architecture.md](docs/architecture.md).

## Run it

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dhia-rek/multi-agent-multi-llm-rag && cd multi-agent-multi-llm-rag
uv sync                                    # pinned dependencies (CPU-only torch on Linux)
uv run python -m scripts.build_index       # one-off: PDFs to FAISS index (~30 s on CPU)
uv run streamlit run app/streamlit_app.py  # http://localhost:8501
```

Without a key the app starts in offline mode. For live models, get a free Gemini key at <https://aistudio.google.com/app/apikey> and put it in `.env` (`cp .env.example .env`, then set `GOOGLE_API_KEY`).

REST API: `uv run uvicorn api.main:app --port 8000` (Swagger UI at `/docs`). Endpoints, settings, pip install and LangSmith tracing: [docs/configuration.md](docs/configuration.md). Docker and Hugging Face Spaces: [deploy/huggingface/DEPLOY.md](deploy/huggingface/DEPLOY.md).

## Screenshots

| Home | RAG inspector |
|---|---|
| ![Home page](docs/img/home.png) | ![RAG inspector](docs/img/rag_inspector.png) |
| **Pipeline run** | **Generated roadmap** |
| ![Pipeline run](docs/img/run_pipeline.png) | ![Roadmap](docs/img/roadmap.png) |

The dashboard also has a routing page with a policy tester, a frameworks reference and a short guide.

## Tests

```bash
uv run pytest -q     # offline: fake embedder, replay client, no network or keys
uv run ruff check .  # lint, including Google-style docstrings
uv run mypy          # type check
uv run interrogate   # docstring coverage, at least 90%
```

GitHub Actions runs lint and unit tests on every push, plus a smoke test that builds the real index, runs the evaluation and calls the API in offline mode.

## Project layout

| Path | What |
|---|---|
| `app/` | Streamlit dashboard and its pages |
| `api/main.py` | FastAPI endpoints |
| `src/agents/` | The six agents and their prompts |
| `src/orchestrator/` | LangGraph graph, streaming runner, trace and cache |
| `src/routing/router.py` | Model tier policy and quota fallback |
| `src/llm_clients/` | Gemini, Ollama, replay and mock clients |
| `src/chunking_embeddings/`, `src/vector_store/`, `src/retrieval/` | RAG pipeline |
| `scripts/` | Index build, evaluation, benchmarks, fixture recording |
| `data/`, `corpus/` | Sample case, replay fixture, evaluation cases, framework PDFs |
| `tests/` | pytest suite |
| `docs/` | Architecture, evaluation, configuration |

## Limitations

- **Example roadmap in the index.** The worked bike shop roadmap is indexed with the framework papers and supplies 63% of retrieved passages on average (100% for the bike shop case), so that case partly retrieves its own answer.
- **Small live evaluation.** Live metrics cover 3 of the 5 cases (free-tier daily quota), one run each, so there is no variance estimate. The evaluator is the same model family it grades.
- **Lexical evaluation.** Field-query precision checks whether a passage names the field; it is a cheap proxy, not human relevance judgement.
- **Small English corpus:** 4 documents, 205 chunks.

## Team

Built by **Dhia** ([@dhia-rek](https://github.com/dhia-rek)) and **Roy** ([@royhayek](https://github.com/royhayek)) as their M2 project for *Multi-LLM Architectures & Eco-Responsible AI* (ECE Paris, MSc AI, 2025–26). After the course, Dhia added the offline mode, the evaluation scripts, the chunking fix, the LangGraph orchestration, tests, CI and the deployment setup (October 2026).

## Licence

Code: [MIT](LICENSE). The PDFs in `corpus/` are third-party publications included for academic use; their copyright stays with their authors (see [corpus/README.md](corpus/README.md)).
