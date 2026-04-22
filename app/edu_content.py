"""Educational copy used across all Streamlit pages.

Centralized so the dashboard explains *what* and *why* in one consistent voice.
"""
from __future__ import annotations

# ─── Agent reference ──────────────────────────────────────────────────────────
AGENTS = {
    "planner": {
        "name": "Planner",
        "icon": "🧭",
        "purpose": "Reads the raw business case and turns it into a structured plan.",
        "what_it_does": (
            "Extracts the company's stakes, objectives, constraints and transformation drivers, "
            "then writes 3–6 short search queries the retrieval step will use to fetch the right "
            "framework passages."
        ),
        "framework_anchor": "Wade 2015 — answers the *Why transform?* question.",
        "task_type": "extraction",
        "complexity": "medium",
        "criticality": "medium",
        "expected_tier": "fast",
        "tier_rationale": "Extraction from natural language is a Flash-tier job — cheap, fast, accurate enough.",
    },
    "framework": {
        "name": "Framework Agent",
        "icon": "📚",
        "purpose": "Classifies retrieved corpus passages by framework axis.",
        "what_it_does": (
            "Maps the RAG passages onto Wade's Why/What/How, Peter's 7 action fields and "
            "Elia's operational pillars (Process / People / Platform / Partners). The output is "
            "the canonical 'reading grid' the downstream agents share."
        ),
        "framework_anchor": "Cross-walks all three corpus frameworks.",
        "task_type": "synthesis",
        "complexity": "medium",
        "criticality": "medium",
        "expected_tier": "fast",
        "tier_rationale": "Structured synthesis over already-retrieved text — Flash handles this well.",
    },
    "canvas_analysis": {
        "name": "Canvas Analysis",
        "icon": "🧮",
        "purpose": "Scores company maturity on the 7 action fields.",
        "what_it_does": (
            "For each of the 7 Digital Transformation Canvas fields (Customer Centricity, New "
            "Technologies, Cloud & Data, Digital Business Development, Process Engineering, "
            "Digital Leadership & Culture, Digital Marketing) it grades current state on 1–5, "
            "lists gaps and opportunities, and flags 2–3 priority fields."
        ),
        "framework_anchor": "Peter 2018 — answers *What to transform?*.",
        "task_type": "reasoning",
        "complexity": "high",
        "criticality": "high",
        "expected_tier": "powerful",
        "tier_rationale": "Multi-axis scoring is reasoning-heavy; we route to the powerful tier.",
    },
    "strategist": {
        "name": "Strategist",
        "icon": "♟️",
        "purpose": "Synthesises a strategic trajectory: purpose, pillars, value, risks.",
        "what_it_does": (
            "Combines the planner's stakes with the canvas analysis to produce a north-star "
            "purpose, the operational pillars, the expected value, the risks, the quick wins "
            "and a list of strategic initiatives anchored back to a framework field."
        ),
        "framework_anchor": "Elia 2024 — strategic purpose, pillars, value, pitfalls.",
        "task_type": "reasoning",
        "complexity": "high",
        "criticality": "high",
        "expected_tier": "powerful",
        "tier_rationale": "Strategic synthesis is the heaviest reasoning step in the pipeline.",
    },
    "roadmap": {
        "name": "Roadmap Generator",
        "icon": "🗺️",
        "purpose": "Turns the strategy into a structured, time-phased roadmap.",
        "what_it_does": (
            "Outputs phases with names, objectives, owners, KPIs, milestones, duration in weeks, "
            "budget tier and dependencies. Each initiative carries a *framework_anchor* so you "
            "can trace it back to Why/What/How or a 7-field name."
        ),
        "framework_anchor": "Peter 2024 — answers *How to transform?*.",
        "task_type": "generation",
        "complexity": "high",
        "criticality": "high",
        "expected_tier": "powerful",
        "tier_rationale": "Final structured deliverable — quality matters most here.",
    },
    "evaluator": {
        "name": "Evaluator",
        "icon": "🔍",
        "purpose": "Critiques the roadmap for coverage, gaps, and risk.",
        "what_it_does": (
            "Independent review pass: checks framework coverage, lists weaknesses, flags "
            "under-treated risks and gives an overall 1–10 score plus concrete recommendations. "
            "Optional — can be skipped to save credits."
        ),
        "framework_anchor": "Spec FR7 — critical evaluation stage.",
        "task_type": "evaluation",
        "complexity": "medium",
        "criticality": "high",
        "expected_tier": "powerful",
        "tier_rationale": "Critique requires the same reasoning depth as the work being critiqued.",
    },
}

# ─── Frameworks reference ─────────────────────────────────────────────────────
FRAMEWORKS = {
    "wade_2015": {
        "title": "Wade 2015 — Why / What / How",
        "source": "Digital Business Transformation: A Conceptual Framework, IMD/Cisco",
        "summary": (
            "Frames transformation as a journey around three guiding questions: "
            "**Why transform?** (drivers), **What to transform?** (objects), "
            "**How to transform?** (path). Reminds us transformation is not a state but a journey."
        ),
        "key_ideas": [
            "Why transform: customer pressure, competition, technology, market evolution, "
            "performance/innovation/agility needs.",
            "What to transform: digital business models, processes, customer relationships, "
            "governance, culture, value creation.",
            "How to transform: structured journey, not a single project.",
        ],
    },
    "peter_2018": {
        "title": "Peter 2018 — Digital Transformation Canvas (7 action fields)",
        "source": "M.K. Peter, Best Practice Reference Guide — marcpeter.com",
        "summary": (
            "Operational canvas with 7 action fields. Comes with workshop-style guiding "
            "questions for each field. Sees DT as a strategic initiative implemented through "
            "several projects: Maturity Analysis → Strategic Analysis → Strategy Development → "
            "Roadmap & Implementation → Change Management & Leadership → Marketing & Continuous Optimisation."
        ),
        "key_ideas": [
            "1. Customer Centricity",
            "2. New Technologies",
            "3. Cloud and Data",
            "4. Digital Business Development",
            "5. Process Engineering",
            "6. Digital Leadership & Culture",
            "7. Digital Marketing",
        ],
    },
    "elia_2024": {
        "title": "Elia et al. 2024 — Strategic & Operational Canvas",
        "source": "Elia, Solazzo, Lerro, Pigni & Tucci, Business Horizons 67",
        "summary": (
            "Academic, more systemic framework: 11 elements grouped in 4 categories — "
            "**Strategy**, **Operational Pillars**, **Value**, **Pitfalls**. "
            "Lets initiatives be compared by budget, timing and risk."
        ),
        "key_ideas": [
            "Strategy: Purpose",
            "Operational Pillars: Process, People, Platform, Partners, Project",
            "Value: Product, Performance, Planet",
            "Pitfalls: Protection, Privacy",
        ],
    },
    "peter_2024": {
        "title": "Peter 2024 — Bikeshop Roadmap Example",
        "source": "M.K. Peter, DigitalProf",
        "summary": (
            "A concrete SME example (Alexandra's Bikeshop). In our project this PDF is treated "
            "as an **input business case**, not as an output template to imitate."
        ),
        "key_ideas": [
            "Used as a realistic business-case stand-in.",
            "Demonstrates how a small Swiss SME might phase its DT.",
        ],
    },
}

# ─── Pipeline blocks ──────────────────────────────────────────────────────────
PIPELINE_BLOCKS = [
    {
        "icon": "📝",
        "name": "User Input",
        "what": "A free-text business case describing the company, its stakes, objectives and constraints.",
        "why": "The system must interpret unstructured natural language — this is what makes LLMs valuable here.",
    },
    {
        "icon": "📚",
        "name": "RAG Retrieval",
        "what": "FAISS searches the indexed corpus for the most relevant chunks using cosine similarity over sentence embeddings.",
        "why": "Anchors reasoning in the provided frameworks instead of letting the LLM hallucinate.",
    },
    {
        "icon": "🤝",
        "name": "Specialized Agents",
        "what": "Six agents with distinct roles, prompts, and expected outputs.",
        "why": "Decomposed reasoning beats one-shot prompting on multi-step tasks; also more explainable and modular.",
    },
    {
        "icon": "🧭",
        "name": "Multi-LLM Routing",
        "what": "Each agent declares task_type / complexity / criticality. The router maps to a tier (fast/powerful/local).",
        "why": "Use a cheap fast model where it's enough, a powerful model where it matters — saves credits without hurting quality.",
    },
    {
        "icon": "🗺️",
        "name": "Final Output",
        "what": "Structured roadmap (phases, initiatives, owners, KPIs, milestones, budget) plus evaluator critique.",
        "why": "The deliverable is decision-grade, not free text.",
    },
]

# ─── Concept glossary ─────────────────────────────────────────────────────────
GLOSSARY = [
    {
        "term": "RAG (Retrieval-Augmented Generation)",
        "plain": (
            "Instead of asking the LLM to answer from memory, we first **search** a corpus "
            "for relevant passages, then **inject** them into the prompt. The model writes its "
            "answer grounded in those passages."
        ),
        "why_here": "Forces every analysis to cite the provided DT frameworks, not the model's general training.",
    },
    {
        "term": "Embedding",
        "plain": (
            "A vector of ~384 numbers that represents the meaning of a piece of text. "
            "Two passages with similar meaning produce similar vectors. We use "
            "`sentence-transformers/all-MiniLM-L6-v2`, which runs locally on the CPU."
        ),
        "why_here": "Lets us 'search by meaning' instead of by keywords across all the framework PDFs.",
    },
    {
        "term": "Vector store / FAISS",
        "plain": (
            "A database optimised for finding the nearest vectors to a query vector. "
            "FAISS (by Meta) is the standard CPU-friendly choice."
        ),
        "why_here": "Holds the embeddings of every corpus chunk so retrieval is sub-millisecond.",
    },
    {
        "term": "Multi-Agent System",
        "plain": (
            "Several specialised agents — each with one role, one prompt, one job — "
            "instead of one monolithic LLM call doing everything."
        ),
        "why_here": "Better reasoning quality, easier debugging, and outputs you can audit step by step.",
    },
    {
        "term": "Multi-LLM Routing",
        "plain": (
            "A small decision layer that picks WHICH model to call for each task: "
            "a cheap/fast model for extraction, a powerful one for strategic reasoning, "
            "and optionally a local one for repetitive tasks."
        ),
        "why_here": "Keeps the bill (and the latency) down without hurting the quality of the strategic stages.",
    },
    {
        "term": "Prompt Mutualization",
        "plain": (
            "All prompt templates live in one file (`src/agents/prompts.py`) "
            "with a shared header. We never rebuild prompts ad hoc per call."
        ),
        "why_here": "Required by the spec (FR9). Easier to tune, less duplication, fewer tokens.",
    },
    {
        "term": "Conditional Execution",
        "plain": "Not every agent runs every time. The evaluator is optional and can be toggled off in the sidebar.",
        "why_here": "When you're iterating on a draft case, skipping eval saves 1 LLM call per run.",
    },
    {
        "term": "Persistent vector DB",
        "plain": (
            "We embed and index the corpus **once**, then save it to `vector_store/index/`. "
            "Subsequent runs reload from disk in milliseconds."
        ),
        "why_here": "Avoids re-embedding 50+ chunks on every Streamlit reload (FR9 optimization).",
    },
]
