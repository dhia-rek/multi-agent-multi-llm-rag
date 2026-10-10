#!/bin/sh
# Container entrypoint: build the FAISS index on first start, then serve the UI.
# Without GOOGLE_API_KEY the app runs in demo mode (DEMO_MODE=auto).
set -e
if [ ! -f vector_store/index/faiss.index ]; then
    echo "Building vector index (first start, ~30 s on CPU)..."
    python -m scripts.build_index
fi
exec streamlit run app/streamlit_app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true
