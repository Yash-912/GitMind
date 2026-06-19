#!/bin/bash
# start.sh — Launch FastAPI and Streamlit in the same container.
# Used by the Hugging Face Spaces Docker runtime.

set -e

echo "==> Starting GitMind API server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# Give the API a moment to initialise before Streamlit tries to call it.
sleep 3

echo "==> Starting Streamlit dashboard on port 7860..."
exec streamlit run interface/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
