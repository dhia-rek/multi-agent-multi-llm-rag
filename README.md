# Digital Transformation Roadmap Generator

**A Multi-Agent RAG System for analyzing business cases and generating structured digital transformation roadmaps**

---

## Overview

This project takes a free-text business case and transforms it into a structured, framework-grounded digital transformation roadmap using multiple AI agents and local RAG (Retrieval-Augmented Generation).

### What It Does

1. **Accepts** your business case (free text description)
2. **Searches** a knowledge base of digital transformation frameworks
3. **Analyzes** using 6 specialized AI agents
4. **Routes** each task to the best LLM model (fast vs powerful)
5. **Generates** a structured, actionable roadmap

---

## Project Team

**Developed by:** Dhia & Roy  
**Course:** ECE MSc AI (M2) — Multi-LLM Architectures & Eco-Responsible AI (2025-26)

---

## Quick Start

### 1. Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Gemini API key (free at https://aistudio.google.com/app/apikey)
```

### 3. Build Vector Index
```bash
# Process PDFs and create FAISS index (one-time setup)
python -m scripts.build_index
```

### 4. Run the Application
```bash
# Start Streamlit UI
streamlit run app/streamlit_app.py
# Opens at http://localhost:8502
```

### 5. (Optional) Run REST API
```bash
# Start FastAPI service in another terminal
uvicorn api.main:app --reload --port 8000
# Swagger docs at http://localhost:8000/docs
```

---

## Architecture

### System Flow

```
User Input
    ↓
Planner Agent (Extract intent)
    ↓
RAG Retrieval (Search knowledge base)
    ↓
Framework Agent (Classify passages)
    ↓
Canvas Agent (Analyze by framework)
    ↓
Strategist Agent (Develop strategy)
    ↓
Roadmap Agent (Generate roadmap)
    ↓
Evaluator Agent (Critique result - optional)
    ↓
Structured Roadmap Output
```

### Key Components

- **RAG Pipeline:** Local FAISS vector store with semantic search
- **Agents:** 6 specialized agents, each with a single responsibility
- **Routing:** Multi-LLM system (fast/powerful/local models)
- **Caching:** SHA-keyed disk cache for optimization
- **UI:** Clean Streamlit interface for end-users
- **API:** FastAPI endpoints for programmatic access

---

## Project Structure

```
├── app/
│   └── streamlit_app.py      # Main user interface
├── api/
│   └── main.py               # FastAPI REST endpoints
├── src/
│   ├── agents/               # 6 AI agents + prompts
│   ├── orchestrator/         # Pipeline orchestration
│   ├── routing/              # LLM model selection
│   ├── llm_clients/          # Gemini, Ollama, Mock
│   ├── retrieval/            # RAG implementation
│   ├── vector_store/         # FAISS index
│   ├── cache/                # Result caching
│   ├── chunking_embeddings/  # Text processing
│   ├── data_ingestion/       # PDF loading
│   └── utils/                # Config & logging
├── scripts/
│   └── build_index.py        # Offline indexing
├── corpus/                   # Reference frameworks (PDFs)
└── vector_store/             # Generated FAISS index
```

---

## Features

### 6 Specialized Agents

1. **Planner** - Extracts key information from business case
2. **Framework** - Maps content to transformation frameworks
3. **Canvas** - Analyzes using 7-field transformation canvas
4. **Strategist** - Develops strategic trajectory
5. **Roadmap** - Generates structured roadmap with phases
6. **Evaluator** - Critiques and validates output (optional)

### Multi-LLM Routing

- **Fast Model** (Gemini Flash-Lite): Extraction, synthesis
- **Powerful Model** (Gemini Flash): Complex reasoning, strategy
- **Local Model** (Ollama): Optional local execution
- **Mock Client** (Deterministic): Testing & fallback

### Optimizations

✓ Persistent FAISS index (load once, reuse)  
✓ Disk cache (same case = instant results)  
✓ Prompt templates (mutualized headers)  
✓ Context capping (6000 chars max)  
✓ Conditional execution (optional evaluator)  
✓ Quota-aware fallback (handles rate limits)  

---

## Usage Examples

### Streamlit UI

1. Enter your business case in the text area
2. Click "Generate Roadmap"
3. View phases, initiatives, and execution details
4. Download results as JSON

### REST API

```bash
# Generate full roadmap
curl -X POST http://localhost:8000/generate-roadmap \
  -H "Content-Type: application/json" \
  -d '{"business_case": "Our bakery wants to go digital..."}'

# Check system status
curl http://localhost:8000/health

# View pipeline configuration
curl http://localhost:8000/pipeline-info
```

---

## Configuration

Create a `.env` file with your settings:

```bash
# LLM Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_FAST_MODEL=gemini-2.5-flash-lite
GEMINI_POWERFUL_MODEL=gemini-2.5-flash

# Optional: Local model
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Requirements

- Python 3.9+
- Google Gemini API key (free tier available)
- FAISS (CPU) for vector search
- Streamlit for UI
- FastAPI for REST endpoints

See `requirements.txt` for full list.

---

## Testing

### Verify Installation
```bash
# Check all dependencies
pip list | grep -E "streamlit|faiss|google|fastapi"

# Test imports
python -c "from src.orchestrator.pipeline import Orchestrator; print('✓ Imports OK')"
```

### Run Sample Analysis
```bash
streamlit run app/streamlit_app.py
# Try with: "We are a small coffee shop wanting to launch online ordering"
```

---

## Performance Notes

- **First run:** ~30-60 seconds (index loading, model initialization)
- **Subsequent runs:** 5-10 seconds (cached results)
- **With real API:** 15-30 seconds (agent execution time)
- **MOCK mode:** <5 seconds (deterministic output)

---

## Troubleshooting

### Index not found
```bash
python -m scripts.build_index
```

### API key errors
- Check `.env` file has valid `GOOGLE_API_KEY`
- Regenerate key at https://aistudio.google.com/app/apikey

### Out of memory
- Reduce `top_k` in config (fewer retrieved passages)
- Use smaller chunk size in `build_index.py`

### Port already in use
```bash
# Use different ports
streamlit run app/streamlit_app.py --server.port=8503
uvicorn api.main:app --port 8001
```

---

## Project Status

✅ Complete implementation of all 11 functional requirements  
✅ All 6 agents working end-to-end  
✅ Multi-LLM routing functional  
✅ Streamlit UI deployed  
✅ REST API available  
✅ Optimization measures implemented  

---

## License

Academic Project - ECE MSc AI (2025-26)

---

## Credits

**Project Team:** Dhia & Roy  
**Supervised by:** ECE Paris - Multi-LLM Architectures Course

For questions or issues, refer to the documentation or contact the development team.
