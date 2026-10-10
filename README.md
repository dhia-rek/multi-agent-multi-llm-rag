# 🧭 Digital Transformation Roadmap Generator

**Six LLM agents and a local RAG pipeline turn a free-text business case into a structured, framework-grounded digital transformation roadmap. Each task is routed to the cheapest model that can handle it.**

[![CI](https://github.com/dhia-rek/multi-agent-multi-llm-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/dhia-rek/multi-agent-multi-llm-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5-4285F4)
![FAISS](https://img.shields.io/badge/vector%20store-FAISS-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)

![Demo: the six agents running on the sample bikeshop case](docs/img/demo.gif)

<sub>Recorded in **demo mode**: retrieval is live, and the agent outputs are replayed from a recorded Gemini run (see [Demo mode](#-demo-mode-no-api-key)).</sub>

> **Live demo:** not deployed yet. The repo is ready for Hugging Face Spaces; see [Deploy](#%EF%B8%8F-deploy).

---

## What it does

You describe a company in plain text, e.g. *"a Swiss bike shop with 8 staff, a basic Shopify site, phone-only repair booking and a CHF 60k budget"*. The system:

1. **Plans**: extracts stakes, objectives and constraints, and writes retrieval queries.
2. **Retrieves**: searches a local FAISS index of three digital-transformation frameworks (Wade 2015, Peter's Digital Transformation Canvas, Elia et al. 2024).
3. **Reasons in stages**: four specialist agents classify the evidence, score maturity on the 7 Canvas action fields, build a strategy and produce a time-phased roadmap.
4. **Critiques**: an optional evaluator agent scores the roadmap against the frameworks.
5. **Routes**: each agent declares its task type, complexity and criticality, and a router sends it to a fast, powerful or local model.

The output is strict JSON (phases, initiatives, owners, KPIs, milestones, budget and framework anchors) plus a full trace showing which model ran each stage, why, and on which passages.

## Engineering highlights

- **Multi-agent orchestration**: six single-purpose agents with typed JSON contracts and a streaming orchestrator, so the UI renders each stage as it finishes.
- **Cost-aware multi-LLM routing**: an explicit policy (`src/routing/router.py`) maps each agent's task type, complexity and criticality to Gemini Flash-Lite, Gemini Flash or Ollama. Per-minute rate limits are retried with the server's suggested delay. If a tier hits its daily quota, it's swapped out so the run still completes.
- **Local RAG**: pypdf → paragraph-aware chunking → MiniLM embeddings on CPU → FAISS cosine index. No API cost for retrieval.
- **Measured, then fixed**: the evaluation harness showed that only 36% of corpus tokens were ever embedded, because whole PDF pages became single chunks. The [chunking fix](#-evaluation) raised that to 99% and more than doubled field-query precision.
- **Works without a key**: demo mode replays a recorded Gemini run's agent outputs over live retrieval, so the app, API, CI and a public demo all run offline.
- **Production basics**: offline pytest suite, ruff, GitHub Actions, `uv.lock`, Dockerfile and Hugging Face Spaces config.
- **Optimisations**: persistent index, SHA-keyed pipeline cache, one shared prompt header, context capped at 6k chars per agent, and an optional evaluator.

---

## 🏗️ Architecture

The pipeline is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` (`src/orchestrator/graph.py`) with a typed state, one node per agent, and Gemini behind `langchain-google-genai`. The router still decides flash-lite vs flash; demo and mock modes plug in as LangChain chat models too.

```mermaid
flowchart LR
    S([▶ start]) --> P[🧭 Planner<br/>flash-lite]
    P -->|retry once if JSON unparsable| P
    P --> R[(📚 Retrieval<br/>FAISS, one batched embedding)]
    R --> F[📚 Framework<br/>flash-lite]
    R --> C[🧮 Canvas analysis<br/>flash]
    F --> J{join}
    C --> J
    J --> ST[♟️ Strategist<br/>flash]
    ST --> RM[🗺️ Roadmap<br/>flash, no thinking]
    RM -->|run_evaluator| E[🔍 Evaluator<br/>flash, thinking 512]
    RM -->|skip| X([■ end])
    E --> X

    classDef par fill:#d1e7dd,stroke:#0f5132
    classDef opt fill:#f8d7da,stroke:#842029
    class F,C par
    class E opt
```

**Real data dependencies** (what each prompt actually consumes):

| Stage | Needs | Can run with |
|---|---|---|
| Planner | business case | – |
| Retrieval | planner's queries | – |
| Framework | planner summary + passages | **Canvas** |
| Canvas | planner summary/objectives/constraints + passages | **Framework** |
| Strategist | plan + canvas + framework | – |
| Roadmap | strategy + canvas | – |
| Evaluator | case + plan + strategy + roadmap | – |

Honest answer: the pipeline is almost entirely sequential. Only **Framework ‖ Canvas** can overlap, and only because the canvas prompt no longer receives the framework agent's output (it used to; the passages it also gets already carry the same evidence). Everything from the strategist on is a strict chain. The other speedups are therefore not parallelism:

- one batched embedding call for all planner queries instead of one per query
- compact JSON between agents (about 2–6% smaller prompts, `python -m scripts.bench_prompts`)
- per-agent thinking budgets: roadmap 0 (it only reshapes a finished strategy into JSON), evaluator 512, others at `GEMINI_THINKING_BUDGET`
- conditional edges: unparsable JSON from a live model re-runs that agent once; quota exhaustion fails over to a mock; missing planner queries fall back to the case text

| Agent | Framework anchor | Job | Tier |
|---|---|---|---|
| 🧭 Planner | Wade 2015 | Extract stakes and objectives, write retrieval queries | fast |
| 📚 Framework | all three | Classify passages by Why/What/How and the 7 action fields | fast |
| 🧮 Canvas analysis | Peter (Canvas) | Score maturity 1–5 per action field, list gaps | powerful |
| ♟️ Strategist | Elia et al. 2024 | Purpose, operational pillars, value, risks, quick wins | powerful |
| 🗺️ Roadmap | Peter (roadmap) | Phases, initiatives, owners, KPIs, milestones, budget | powerful |
| 🔍 Evaluator | all three | Critique, coverage check, score 1–10 | powerful |

<details>
<summary><b>Routing policy</b></summary>

```mermaid
flowchart LR
    A[Agent declares<br/>task_type · complexity · criticality] --> R{Router}
    R -->|reasoning / generation / evaluation<br/>or complexity=high or criticality=high| P[🧠 Powerful<br/>gemini-2.5-flash]
    R -->|criticality=low and Ollama configured| L[🏠 Local<br/>Ollama]
    R -->|otherwise| F[⚡ Fast<br/>gemini-2.5-flash-lite]
    P -.daily quota hit.-> M[🎭 Mock fallback]
    F -.daily quota hit.-> M
```

Each agent sets three class attributes (`src/agents/*.py`); `Router.decide()` turns them into a tier and a human-readable reason that the UI shows.

</details>

<details>
<summary><b>RAG pipeline</b></summary>

```mermaid
flowchart LR
    PDF[corpus/*.pdf] --> L[pypdf loader]
    L --> CH[Chunker<br/>≤900 chars, 150 overlap]
    CH --> EMB[MiniLM-L6-v2<br/>local CPU]
    EMB --> IDX[(FAISS<br/>inner product on<br/>normalised vectors)]
    Q[Agent query] --> EMBQ[MiniLM] --> S[top-k search] --> CTX[Passages with<br/>framework · title · page · score]
    IDX --> S
```

</details>

### ⚡ Old vs new orchestration

Same bikeshop case, `gemini-2.5-flash-lite` (planner, framework) and `gemini-2.5-flash` (the rest).

| | Old (sequential, main) | New (LangGraph) | How measured |
|---|---|---|---|
| Stages on the critical path | 7 | 6 (framework ‖ canvas) | graph structure, tested |
| Retrieval, 4 queries (median of 15, CPU) | 29.9 ms | 11.1 ms | measured, batched embedding |
| Strategist / roadmap / evaluator prompt size | 8356 / 7224 / 9980 chars | 7859 / 6845 / 9798 chars (−6% / −5% / −2%) | measured, `scripts.bench_prompts` |
| Canvas prompt | 2095 prompt tokens | smaller (no framework insights) | old: recorded run; new: not measured live |
| Thinking tokens, roadmap + evaluator | 756 + 776 | ≤ 0 + 512 (caps) | budgets set in code; actual use not measured live |
| End-to-end wall clock | **47.6 s** (recorded live run, 2026-10-10: 2.0+1.9+2.1+9.4+7.1+15.1+10.0) | **≈ 43 s, projection** | framework's 2.1 s hides behind canvas; roadmap without thinking is the likely bigger win |
| Total tokens | 10 632 prompt + 6 710 output + 3 159 thinking | not measured | – |

**Caveat, read this:** a live new-vs-old Gemini comparison could not be completed. The free-tier daily cap (20 requests per model per day) on `gemini-2.5-flash` and then `flash-lite` was reached by the first benchmark runs, so later runs fell back to the mock and say nothing about speed. The old figures come from the recorded live run in the demo fixture; the new end-to-end figure is a projection from those stage timings, not a measurement. Rerun when quota resets: `python -m scripts.bench_live old` on `main` and `python -m scripts.bench_live new` on this branch (about 6 calls each).

---

## 📊 Evaluation

`python -m scripts.evaluate` runs offline over 5 SME business cases (`data/eval/cases.jsonl`).

**Chunking fix, before and after** (same corpus, same embedder):

| Metric | Before | After |
|---|---|---|
| Chunks | 52 | 205 |
| Corpus tokens inside the embedder's 256-token window | 36% | **99%** |
| Field-query precision@3 (7 Canvas fields; share of top-3 passages that discuss the field) | 0.19 | **0.48** |
| Median retrieval latency (CPU) | 17 ms | 20 ms |

pypdf output rarely contains blank lines, so the chunker treated each page as one paragraph. Chunks reached several thousand characters, and MiniLM silently truncated them, so most of each page was invisible to retrieval.

**Per-case retrieval** (top-6, query = business case text):

| Case | Framework papers in top-6 (of 3) | Share of passages from the example roadmap |
|---|---|---|
| alexandra_bikeshop | 0 | 1.00 |
| family_bakery | 2 | 0.50 |
| regional_logistics | 2 | 0.33 |
| dental_clinic_group | 2 | 0.50 |
| industrial_parts_distributor | 1 | 0.83 |

The worked bikeshop roadmap (`corpus/bikeshop_roadmap/`) is indexed alongside the framework papers and dominates retrieval, especially for the bikeshop case itself. See [Limitations](#%EF%B8%8F-limitations).

**Live pipeline metrics** (`python -m scripts.evaluate --live`, needs `GOOGLE_API_KEY`). Run on 2026-10-10 with `gemini-2.5-flash-lite` (fast tier) and `gemini-2.5-flash` (powerful tier), free tier:

| Case | Agents with valid JSON | Canvas priority fields in expected set | Evaluator score (1–10) | Latency (s) |
|---|---|---|---|---|
| alexandra_bikeshop | 6/6 | 0.67 | 6 | 47.5 |
| family_bakery | 6/6 | 0.67 | 6 | 45.6 |
| regional_logistics | 6/6 | 0.67 | 6 | 42.7 |
| dental_clinic_group | not run: daily quota | | | |
| industrial_parts_distributor | not run: daily quota | | | |
| **3 cases** | **18/18** | **0.67** | **6.0** | **45.3 mean** |

The free-tier daily cap on `gemini-2.5-flash` ran out during the fourth case, after 16 successful requests that day. The router fell back to the mock as designed, so those two rows measure nothing and are left out. Latency is the sum of stage timings; in the recorded bikeshop run the four powerful-tier stages (canvas, strategist, roadmap, evaluator) took 87% of it.

Before the thinking-budget fix, this table would have read 2/6: `gemini-2.5-flash` thinks by default, and thinking tokens count against `max_output_tokens`. With the old 2048 limit it spent ~1960 tokens thinking and cut every powerful-tier answer off after ~80 tokens. The powerful tier now gets `GEMINI_THINKING_BUDGET` (1024) on top of a 4096-token answer budget, and the fast tier runs with thinking off.

---

## 🚀 Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dhia-rek/multi-agent-multi-llm-rag && cd multi-agent-multi-llm-rag
uv sync                                   # installs pinned deps (CPU-only torch on Linux)
uv run python -m scripts.build_index      # one-off: PDFs → FAISS index (~30 s on CPU)
uv run streamlit run app/streamlit_app.py # → http://localhost:8501
```

That's enough to use the app: with no key it starts in **demo mode**. For live models, get a free Gemini key at <https://aistudio.google.com/app/apikey>, then:

```bash
cp .env.example .env    # set GOOGLE_API_KEY=...
```

<details>
<summary>Using pip instead of uv</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned export of uv.lock
python -m scripts.build_index
streamlit run app/streamlit_app.py
```

</details>

### REST API

```bash
uv run uvicorn api.main:app --port 8000   # Swagger UI at http://localhost:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, configured models, index ready, `demo_mode` |
| GET | `/pipeline-info` | Models per tier, embedding model, chunking, top-k |
| POST | `/analyze` | Planner → RAG → Framework → Canvas only |
| POST | `/generate-roadmap` | Full pipeline: roadmap, optional evaluation, trace |

```bash
curl -s -X POST localhost:8000/generate-roadmap -H 'Content-Type: application/json' \
  -d '{"business_case": "Family bakery, 12 staff, wants online pre-orders and a loyalty app.", "run_evaluator": false}'
```

### 🎭 Demo mode (no API key)

| `DEMO_MODE` | Behaviour |
|---|---|
| `auto` (default) | Demo mode when `GOOGLE_API_KEY` is empty |
| `on` | Always demo mode, even with a key |
| `off` | Never: without a key, agents return placeholder mock text |

In demo mode, retrieval runs for real, while the six agents replay [`data/demo/alexandra_bikeshop.fixture.json`](data/demo/alexandra_bikeshop.fixture.json). **That fixture was recorded from a live Gemini run on 2026-10-10** (`gemini-2.5-flash-lite` for planner and framework, `gemini-2.5-flash` for the other four agents). Outputs are verbatim, with per-stage timings and token usage in its `recorded` block. Regenerate it with `python -m scripts.record_demo` (6 Gemini calls). Any other business case will still show bikeshop outputs, and the UI says so. Demo traces are cached separately from live ones.

### 🔭 LangSmith tracing (optional)

Tracing is off unless you set the standard LangChain variables, for example in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU region
LANGSMITH_PROJECT=multi-agent-rag
```

Each pipeline run becomes one `LangGraph` trace with a span per agent node, the Gemini calls, prompts and token usage; the parallel framework/canvas branches show side by side. Works in demo mode too (the fixture client is traced as a chat model). Nothing else to configure.

### Configuration

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

---

## 🖼️ Screenshots

| Home: architecture overview | RAG inspector |
|---|---|
| ![Home page](docs/img/home.png) | ![RAG inspector](docs/img/rag_inspector.png) |
| **Live pipeline run** | **Generated roadmap** |
| ![Run pipeline](docs/img/run_pipeline.png) | ![Roadmap](docs/img/roadmap.png) |

The dashboard also has a **Multi-LLM Routing** page with a live policy tester, a **Frameworks** reference and a **How It Works** glossary.

---

## 🧪 Tests and CI

```bash
uv run pytest -q        # offline: fake embedder, demo client, no network or keys
uv run ruff check .
```

GitHub Actions runs two jobs on every push and PR: **lint + unit tests**, and a **smoke test** that builds the real index, runs the evaluation and calls the API in demo mode.

## ☁️ Deploy

`Dockerfile` + `deploy/start.sh` produce a CPU image that builds the index on first boot and serves Streamlit on port 7860. That's the layout Hugging Face Spaces (Docker SDK) expects. Step-by-step instructions and the secrets table are in [`deploy/huggingface/DEPLOY.md`](deploy/huggingface/DEPLOY.md).

```bash
docker build -t dt-roadmap . && docker run -p 7860:7860 -e GOOGLE_API_KEY=... dt-roadmap
```

---

## 📁 Project layout

```
app/                    Streamlit dashboard: home + 5 pages, teaching copy in edu_content.py
api/main.py             FastAPI service
src/
  agents/               6 agents, shared base (routing call, lenient JSON parsing), prompts.py
  orchestrator/         graph.py: LangGraph StateGraph; pipeline.py: run() and run_streaming(), disk-cached
  routing/router.py     tier policy + quota failover
  llm_clients/          gemini (langchain-google-genai) · ollama · demo (fixture) · mock, all as LangChain chat models
  data_ingestion/       pypdf loader
  chunking_embeddings/  chunker + sentence-transformers embedder
  vector_store/         FAISS build / save / load / search
  retrieval/            top-k retriever with framework filter
  cache/                SHA-keyed disk cache
scripts/                build_index.py · evaluate.py · record_demo.py · bench_live.py · bench_prompts.py
data/                   sample case · demo fixture · eval cases
corpus/                 reference PDFs (third-party, see corpus/README.md)
tests/                  pytest suite
deploy/                 container entrypoint + Hugging Face Space card
```

## 🧰 Stack

Python · LangChain + LangGraph · Google Gemini 2.5 (`langchain-google-genai`) · Ollama (optional) · sentence-transformers (MiniLM-L6-v2) · FAISS · pypdf · Streamlit · FastAPI · pydantic · tenacity · uv · pytest · ruff · GitHub Actions · Docker

## ⚠️ Limitations

- **Example roadmap in the index.** The worked bikeshop roadmap is indexed with the framework papers and supplies 63% of retrieved passages on average (100% for the bikeshop case). That's useful as an exemplar, but it also means the sample case partly retrieves its own answer.
- **Small live evaluation.** Live metrics cover 3 of the 5 cases (free-tier daily quota), from one run each, so there's no variance estimate. The evaluator is the same model family it grades.
- **Lexical evaluation.** Field-query precision checks whether a passage names the field. It's a cheap proxy, not human relevance judgement.
- **English PDFs only, small corpus** (4 documents, 205 chunks).

---

## 👥 Team and project context

Built by **Dhia** ([@dhia-rek](https://github.com/dhia-rek)) and **Roy** ([@royhayek](https://github.com/royhayek)) as their M2 project for *Multi-LLM Architectures & Eco-Responsible AI* (ECE Paris, MSc AI, 2025–26). The brief: an AI system that turns a business case into a digital transformation roadmap, with RAG over provided frameworks, specialised agents, multi-LLM routing, cost optimisations, a dashboard and a REST API.

Dhia and Roy designed and built the system together during the course. After the course, a showcase pass (October 2026) added the offline demo mode, evaluation harness, chunking fix, Gemini SDK migration, tests, CI, packaging and deployment setup.

## 📄 License

Code: [MIT](LICENSE). The PDFs in `corpus/` are third-party publications included for academic use; their copyright stays with their authors (see [`corpus/README.md`](corpus/README.md)).
