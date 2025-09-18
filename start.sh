#!/usr/bin/env sh
set -e

: "${DATA_DIR:=/data}"
mkdir -p "$DATA_DIR"

# Start FastAPI backend
uvicorn src.backend:app --host 0.0.0.0 --port 8000 &

# Start Streamlit UI
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false streamlit run src/frontend.py --server.address 0.0.0.0 --server.port 8501


