#!/bin/bash
# start.sh — Launch FastAPI and Streamlit in the same container.
# Used by the Hugging Face Spaces Docker runtime.
#
# Streamlit starts only after the API answers /health. If either process dies
# the other is stopped and the script exits non-zero (so the platform restarts
# the container instead of serving a half-dead app).

set -uo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-7860}"
API_WAIT_S="${API_WAIT_S:-120}"

pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting GitMind API server on port ${API_PORT}..."
uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" --workers 1 &
api_pid=$!
pids+=("$api_pid")

echo "==> Waiting for the API (/health) ..."
healthy=0
for _ in $(seq 1 "${API_WAIT_S}"); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "ERROR: API process exited during startup." >&2
        exit 1
    fi
    sleep 1
done
if [ "$healthy" -ne 1 ]; then
    echo "ERROR: API did not become healthy within ${API_WAIT_S}s." >&2
    exit 1
fi

echo "==> Starting Streamlit dashboard on port ${UI_PORT}..."
streamlit run interface/streamlit_app.py \
    --server.port "${UI_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
pids+=("$!")

# Block until the first of the two exits; neither is supposed to.
wait -n
status=$?
echo "ERROR: a service exited (status ${status}); shutting down." >&2
if [ "$status" -eq 0 ]; then
    status=1
fi
exit "$status"
