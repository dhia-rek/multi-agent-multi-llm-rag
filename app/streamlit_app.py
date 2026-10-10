"""
Digital Transformation Roadmap Generator
Student Project - ECE MSc AI (2025-26)
Multi-LLM Architectures & Eco-Responsible AI

Project Team: Dhia & Roy
Course: ECE Paris - Multi-LLM Architectures

Description:
    Interactive Streamlit interface for analyzing business cases
    and generating structured digital transformation roadmaps.
    
Features:
    - Free-text business case input
    - Multi-agent analysis pipeline
    - Framework-grounded recommendations
    - Real-time progress tracking
    - Result download (JSON format)
"""

import sys
import json
from pathlib import Path

import streamlit as st

# Setup path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import settings
from src.vector_store.faiss_store import FaissStore
from src.orchestrator.pipeline import Orchestrator

# Page configuration
st.set_page_config(page_title="DT Roadmap Generator", layout="wide")

# ===== SIDEBAR =====
with st.sidebar:
    st.title("DT Roadmap Generator")
    st.write("Transform your business case into a digital transformation roadmap")
    st.divider()
    
    # Check system status
    st.subheader("System Status")
    
    try:
        info = FaissStore(settings.index_dir).info()
        if info:
            chunks = info.get('n_chunks', 0)
            st.success(f"Vector index ready: {chunks} chunks loaded")
        else:
            st.error("Vector index not found")
    except Exception as e:
        st.error(f"Vector index error: {str(e)}")

    # Check API key
    if settings.google_api_key:
        st.success("Gemini API key is configured")
    else:
        st.warning("No API key - running in MOCK mode")
    
    st.caption(f"LLM Model: {settings.gemini_fast_model}")


# ===== MAIN PAGE =====
st.title("Digital Transformation Roadmap Generator")

st.write("""
This tool analyzes your business case and generates a digital transformation roadmap.

**How it works:**
1. Enter your company details and transformation needs
2. The system searches our knowledge base of frameworks
3. Multiple AI agents analyze your situation
4. You get a structured roadmap with phases and initiatives
""")

st.divider()

# Input form
st.subheader("Enter Your Business Case")

business_case = st.text_area(
    "Describe your company, goals, budget, and constraints:",
    height=150,
    placeholder="Example: We are a bakery with 5 employees. We want to launch an online store and implement a customer loyalty program. Budget: 15,000 EUR. Timeline: 6 months."
)

# Options
col1, col2 = st.columns([3, 1])
with col1:
    run_button = st.button("Generate Roadmap", type="primary", use_container_width=True)
with col2:
    include_evaluation = st.checkbox("Evaluate Result", value=True)

st.divider()

# ===== RESULTS =====
if run_button:
    # Validate input
    if not business_case.strip():
        st.error("Please enter a business case")
        st.stop()
    
    # Create orchestrator and run pipeline
    try:
        orchestrator = Orchestrator()
        status = st.empty()
        info = st.empty()
        
        # Run analysis
        for stage_name, trace in orchestrator.run_streaming(business_case, run_evaluator=include_evaluation):
            if stage_name == "cached":
                status.info("Loaded from cache")
            elif stage_name != "done":
                stage_display = stage_name.replace("_", " ").title()
                status.info(f"Running: {stage_display}...")
            else:
                status.success("Analysis complete!")
                result = trace
        
        # Display results
        st.divider()
        st.subheader("Your Digital Transformation Roadmap")
        
        # Get roadmap from result
        if result.roadmap and result.roadmap.output:
            roadmap = result.roadmap.output
            
            if isinstance(roadmap, dict):
                # Display phases
                if "phases" in roadmap:
                    st.markdown("### Phases")
                    for phase in roadmap["phases"]:
                        with st.expander(f"{phase.get('name')} - {phase.get('duration')}"):
                            for key, value in phase.items():
                                if key != "name":
                                    st.write(f"**{key}**: {value}")
                
                # Display initiatives
                if "initiatives" in roadmap:
                    st.markdown("### Initiatives")
                    for initiative in roadmap["initiatives"]:
                        with st.expander(f"{initiative.get('name')} (Priority: {initiative.get('priority')})"):
                            for key, value in initiative.items():
                                if key != "name":
                                    st.write(f"**{key}**: {value}")
            
            # Show execution details
            st.divider()
            with st.expander("Show Execution Details"):
                st.write("**Processing Time:**")
                for stage, time in result.timings.items():
                    st.write(f"- {stage}: {time}s")
                
                st.write("**Models Used:**")
                for stage, model in result.model_per_stage.items():
                    st.write(f"- {stage}: {model}")
            
            # Download button
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Roadmap (JSON)",
                    json.dumps(roadmap, indent=2, ensure_ascii=False),
                    "roadmap.json",
                    "application/json"
                )
            with col2:
                st.download_button(
                    "Download Full Analysis (JSON)",
                    json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
                    "full_analysis.json",
                    "application/json"
                )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Show Error Details"):
            import traceback
            st.code(traceback.format_exc())

st.divider()
st.caption("Built with: Multi-Agent RAG | FAISS Vector Store | Multi-LLM Routing")
