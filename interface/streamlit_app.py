"""interface/streamlit_app.py — GitMind dashboard.

In production (HF Spaces) this app is a thin REST client that calls the
FastAPI backend running in the same container on port 8000.
Set the env var GITMIND_API_URL to point to a different backend.
"""
from __future__ import annotations

import os
import time

import httpx
import streamlit as st

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

# When running inside HF Spaces the FastAPI server starts on port 8000
# and Streamlit on port 7860.  GITMIND_API_URL defaults to localhost.
API_URL = os.getenv("GITMIND_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")  # optional — matches the FastAPI secret

_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(base_url=API_URL, timeout=180.0) as client:
        resp = client.post(path, json=payload, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


def _get(path: str) -> dict:
    with httpx.Client(base_url=API_URL, timeout=30.0) as client:
        resp = client.get(path, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------ #
# Page layout                                                          #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="GitMind — Codebase Archaeology",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar ----
with st.sidebar:
    st.image(
        "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        width=40,
    )
    st.title("GitMind")
    st.caption("Ask **why** questions about your codebase history.")

    st.divider()

    # Health check
    with st.expander("🔌 Backend status", expanded=False):
        if st.button("Check"):
            try:
                data = _get("/health")
                if data.get("status") == "ok":
                    st.success("API is healthy ✅")
                else:
                    st.warning("API degraded ⚠️")
                st.json(data)
            except Exception as exc:
                st.error(f"Cannot reach API: {exc}")

    st.divider()

    # Ingestion
    st.subheader("📥 Ingest Repository")
    repo_path = st.text_input("Local repo path", value=".")
    github_repo = st.text_input("GitHub repo (owner/name)", value="")
    max_commits = st.number_input("Max commits (0 = all)", value=0, min_value=0)

    if st.button("Start Ingestion"):
        with st.spinner("Triggering ingestion…"):
            try:
                payload: dict = {"repo_path": repo_path}
                if github_repo:
                    payload["github_repo"] = github_repo
                if max_commits:
                    payload["max_commits"] = max_commits
                resp = _post("/api/v1/ingest", payload)
                st.success(f"Ingestion started — task `{resp.get('task_id', '?')}`")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

# ---- Main area ----
st.title("🧠 GitMind — Codebase Archaeology")
st.write(
    "Ask natural language questions about **why** your codebase is the way it is. "
    "GitMind reconstructs institutional knowledge from git history, PRs, and issues."
)

query = st.text_area(
    "Your question",
    placeholder="e.g. Why does the auth module use JWT instead of session cookies?",
    height=80,
)

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    mode = st.selectbox(
        "Answer mode",
        options=["direct", "memo", "blame", "risk"],
        format_func=lambda x: {
            "direct": "📝 Direct Q&A",
            "memo": "📋 Decision Memo",
            "blame": "👤 Blame Map",
            "risk": "⚠️ Risk Report",
        }[x],
    )
with col2:
    top_k = st.slider("Evidence chunks (top-k)", min_value=3, max_value=20, value=12)
with col3:
    limit = st.slider("Retrieval candidates", min_value=10, max_value=100, value=40)

run = st.button("🔍 Answer", type="primary", disabled=not query.strip())

if run and query.strip():
    with st.spinner("Retrieving and generating answer…"):
        t0 = time.time()
        try:
            data = _post(
                "/api/v1/query",
                {"query": query, "mode": mode, "top_k": top_k, "limit": limit},
            )
            elapsed = time.time() - t0

            st.success(f"Done in {elapsed:.1f}s  •  Model: `{data.get('model', '?')}`")

            # Answer
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            # Evidence table
            evidence = data.get("evidence", [])
            if evidence:
                st.subheader(f"Evidence ({len(evidence)} chunks)")
                rows = [
                    {
                        "score": f"{e['score']:.4f}",
                        "source": e["source"],
                        "doc_type": e["doc_type"],
                        "doc_id": e["doc_id"],
                        "author": e["author"],
                        "timestamp": e["timestamp"][:10] if e["timestamp"] else "",
                        "snippet": e["snippet"],
                    }
                    for e in evidence
                ]
                st.dataframe(rows, use_container_width=True)

        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
