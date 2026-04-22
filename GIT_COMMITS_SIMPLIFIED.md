# Simplified Git Commit History

## Overview
The commit history has been simplified to use basic, clear messages instead of complex conventional commit format (feat(), docs(), refactor(), etc.).

## Current Simplified Commit Structure

All commits now follow a simple pattern:
- Action: `Load`, `Add`, `Update`, `Setup`, `Refactor`, `Simplify`
- Component: `PDFs`, `embeddings`, `vector store`, `retrieval`, etc.

## Commit History (Simplified)

```
2c8124c Simplify UI                          (Most Recent)
3e0a7b0 Update docs
795fe84 Refactor pages
3041a3e Add FastAPI
aede419 Update README
c66cc4b Add quota fallback
7034a90 Add framework pages
5df27e0 Add routing page
ed32bf7 Add RAG page
e3c2ffd Add pipeline
5622ce0 Update home
df5c3b5 Add app config
f111a41 Add Streamlit
df5c3b5 Add app config
5622ce0 Update home
e3c2ffd Add pipeline
ed32bf7 Add RAG page
5df27e0 Add routing page
7034a90 Add framework pages
c66cc4b Add quota fallback
aede419 Update README
3041a3e Add FastAPI
795fe84 Refactor pages
3e0a7b0 Update docs
566fe3e Add orchestrator
3d15b5c Add caching
2ca8f80 Add Roadmap agent
df30d48 Add Canvas agent
21f2dd9 Add Planner agent
4d1923f Add prompts
3e479c6 Add base agent
afe80f1 Add routing
d1636f5 Add Ollama
60903d1 Add Gemini
582e048 Add LLM base
d3b71eb Add retrieval
854eaa6 Add indexing
a74b3ba Setup vector store
6b249b5 Add embeddings
08c2416 Chunk documents
a24e9a5 Load PDFs                           (Oldest)
```

## Commits by Category

### Data Pipeline
- `a24e9a5`: Load PDFs
- `08c2416`: Chunk documents
- `6b249b5`: Add embeddings
- `a74b3ba`: Setup vector store
- `854eaa6`: Add indexing

### Retrieval & Search
- `d3b71eb`: Add retrieval

### LLM Clients
- `582e048`: Add LLM base
- `60903d1`: Add Gemini
- `d1636f5`: Add Ollama

### Routing & Decision Layer
- `afe80f1`: Add routing

### Agents
- `3e479c6`: Add base agent
- `4d1923f`: Add prompts
- `21f2dd9`: Add Planner agent
- `df30d48`: Add Canvas agent
- `2ca8f80`: Add Roadmap agent

### Orchestration
- `3d15b5c`: Add caching
- `566fe3e`: Add orchestrator

### User Interface
- `f111a41`: Add Streamlit
- `df5c3b5`: Add app config
- `5622ce0`: Update home
- `e3c2ffd`: Add pipeline
- `ed32bf7`: Add RAG page
- `5df27e0`: Add routing page
- `7034a90`: Add framework pages

### API & Documentation
- `c66cc4b`: Add quota fallback
- `aede419`: Update README
- `3041a3e`: Add FastAPI
- `3e0a7b0`: Update docs
- `795fe84`: Refactor pages

### Final Cleanup
- `2c8124c`: Simplify UI

## Why These Simple Messages?

1. **Clarity**: Easy to understand what changed without learning conventional commit format
2. **Student-friendly**: More natural and less formal
3. **Easy to scan**: Quick overview of project progression
4. **Professional**: Still clear and organized, just simpler

## Project Flow

The commits tell the story of building the system:
1. First: Load and process PDFs → setup vector store
2. Then: Add embeddings and retrieval  
3. Then: Add LLM clients (Gemini, Ollama)
4. Then: Add routing logic
5. Then: Build agents (base → specific agents)
6. Then: Add orchestration and caching
7. Then: Build Streamlit UI with all pages
8. Finally: Add API and clean up

Each step is atomic and focused on a single feature.
