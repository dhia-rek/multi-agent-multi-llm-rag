"""FR11 — REST API layer.

Exposes the pipeline as a small HTTP service so the dashboard (or any other
client) can call the AI engine without loading the model, the vector store
or the prompts directly. This implements the Front / API / Backend split
recommended in the spec (§9).

Endpoints
---------
    GET  /health             -> liveness check + which models are configured
    GET  /pipeline-info      -> pipeline config (models per tier, top-k, etc.)
    POST /analyze            -> run planner + RAG + framework + canvas only
    POST /generate-roadmap   -> run the full pipeline end-to-end

Run locally:
    uvicorn api.main:app --reload --port 8000
    -> docs at http://localhost:8000/docs (FastAPI auto-generated Swagger)
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator.pipeline import Orchestrator
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="DT Roadmap API",
    description=(
        "Multi-Agent, Multi-LLM RAG service that turns a free-text business case "
        "into a structured digital transformation roadmap."
    ),
    version="1.0.0",
)

# One orchestrator per process — loads the FAISS index and the prompt templates
# only once, then reuses them for every request.
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        log.info("Bootstrapping Orchestrator (first request)...")
        _orchestrator = Orchestrator()
    return _orchestrator


# ─── Request / response models ────────────────────────────────────────────────
class BusinessCaseIn(BaseModel):
    business_case: str = Field(
        ...,
        min_length=20,
        description="Free-text description of the company, its stakes, objectives and constraints.",
    )
    run_evaluator: bool = Field(
        True,
        description="Whether to run the optional evaluator agent (FR7). Set false to save 1 LLM call.",
    )


class HealthOut(BaseModel):
    status: str
    gemini_configured: bool
    fast_model: str
    powerful_model: str
    local_model: str | None
    vector_index_ready: bool


class PipelineInfoOut(BaseModel):
    fast_model: str
    powerful_model: str
    local_model: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    """Liveness + minimal config check. Lets the dashboard show a green/red light."""
    index_ready = (settings.index_dir / "faiss.index").exists()
    return HealthOut(
        status="ok",
        gemini_configured=bool(settings.google_api_key),
        fast_model=settings.gemini_fast_model,
        powerful_model=settings.gemini_powerful_model,
        local_model=settings.ollama_model,
        vector_index_ready=index_ready,
    )


@app.get("/pipeline-info", response_model=PipelineInfoOut, tags=["meta"])
def pipeline_info() -> PipelineInfoOut:
    """Active configuration — useful to show 'which model runs where' in any client."""
    return PipelineInfoOut(
        fast_model=settings.gemini_fast_model,
        powerful_model=settings.gemini_powerful_model,
        local_model=settings.ollama_model,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


@app.post("/analyze", tags=["pipeline"])
def analyze(payload: BusinessCaseIn) -> dict[str, Any]:
    """Run the analysis stages only: Planner → RAG → Framework → Canvas.

    Returns the canvas maturity grading plus the retrieval trace — useful as
    a quick diagnostic before generating the full roadmap.
    """
    if not (settings.index_dir / "faiss.index").exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run `python -m scripts.build_index` first.",
        )

    orch = get_orchestrator()
    trace = orch.run(payload.business_case, run_evaluator=False)

    return {
        "plan": trace.plan.output if trace.plan else None,
        "retrieval": [
            {
                "framework": p.chunk.framework,
                "title": p.chunk.title,
                "page": p.chunk.page,
                "score": round(p.score, 4),
            }
            for p in trace.retrieval
        ],
        "framework": trace.framework.output if trace.framework else None,
        "canvas": trace.canvas.output if trace.canvas else None,
        "models_used": trace.model_per_stage,
        "timings_seconds": trace.timings,
    }


@app.post("/generate-roadmap", tags=["pipeline"])
def generate_roadmap(payload: BusinessCaseIn) -> dict[str, Any]:
    """Run the full pipeline and return the structured roadmap (FR6) + optional evaluation (FR7)."""
    if not (settings.index_dir / "faiss.index").exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run `python -m scripts.build_index` first.",
        )

    orch = get_orchestrator()
    try:
        trace = orch.run(payload.business_case, run_evaluator=payload.run_evaluator)
    except Exception as e:  # surfacing as a 500 with a readable message
        log.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e

    return trace.as_dict()
# REST API - May 6
