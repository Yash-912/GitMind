"""interface/streamlit_app.py — GitMind dashboard.

A thin REST client for the FastAPI backend (GITMIND_API_URL, default
http://localhost:8000). All logic that can be tested without Streamlit lives
in interface/api_client.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interface.api_client import ApiError, GitMindClient, build_query_payload  # noqa: E402

API_URL = os.getenv("GITMIND_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")  # optional: matches the FastAPI secret
ACTIVE_JOB_STATES = {"queued", "running"}


@st.cache_resource
def _client() -> GitMindClient:
    return GitMindClient(API_URL, API_KEY)


def _source_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "chunk": s["chunk_id"][:12],
            "doc_type": s["doc_type"],
            "doc_id": s["doc_id"],
            "author": s["author"],
            "timestamp": (s["timestamp"] or "")[:19],
            "commit": s["commit_sha"][:10],
            "path": s["path"],
            "lines": f"{s['line_start']}-{s['line_end']}" if s.get("line_start") else "",
            "url": s["source_url"],
            "score": s["score"],
            "snippet": s["snippet"],
        }
        for s in items
    ]


st.set_page_config(page_title="GitMind — Codebase Archaeology", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("GitMind")
    st.caption("Ask **why** questions about your codebase history.")
    st.divider()

    ready_repos: list[str] = []
    with st.expander("🔌 Backend readiness", expanded=False):
        if st.button("Check"):
            try:
                report = _client().ready()
                (st.success if report["status"] == "ready" else st.warning)(
                    f"API status: {report['status']}"
                )
                for item in report["repos"]:
                    st.write(f"**{item['repo']}** — {'ready' if item['ready'] else 'NOT ready'}")
                    st.json(item)
                if report["degraded"]:
                    st.warning("Degraded: " + "; ".join(report["degraded"]))
            except ApiError as exc:
                st.error(f"Cannot read readiness: {exc.message}")
    try:
        ready_repos = [r["repo"] for r in _client().ready()["repos"] if r["ready"]]
    except ApiError:
        pass

    st.divider()
    st.subheader("📥 Ingest repository")
    repo_path = st.text_input("Local repo path", value=".")
    github_repo = st.text_input("GitHub repo (owner/name)", value="")
    max_commits = st.number_input("Max commits (0 = all)", value=0, min_value=0)
    if st.button("Start ingestion"):
        try:
            resp = _client().start_ingest(repo_path, github_repo, int(max_commits))
            st.session_state["job_id"] = resp["job_id"]
            st.session_state["job_repo"] = resp.get("repo")
            st.success(f"Queued job `{resp['job_id'][:8]}` for {resp.get('repo')}")
        except ApiError as exc:
            st.error(f"Ingestion not started ({exc.status_code}): {exc.message}")

    @st.fragment(run_every="2s")
    def job_panel() -> None:
        job_id = st.session_state.get("job_id")
        if not job_id:
            return
        try:
            job = _client().job(job_id)
        except ApiError as exc:
            st.error(f"Cannot read job: {exc.message}")
            return
        st.markdown(f"**Job** `{job['id'][:8]}` · {job['repo']} · **{job['status']}**")
        if job["progress"]:
            st.caption(job["progress"])
        if job["status"] in ACTIVE_JOB_STATES:
            st.progress(0.5 if job["status"] == "running" else 0.0)
            if st.button("Cancel job", key="cancel_job"):
                try:
                    _client().cancel(job_id)
                except ApiError as exc:
                    st.error(exc.message)
        elif job["status"] == "succeeded":
            st.success("Done — the new index generation is live.")
        else:
            st.error(f"{job['status']}: {job['error'] or 'no details'}")

    job_panel()

    with st.expander("Recent jobs"):
        try:
            st.dataframe(
                [
                    {k: j[k] for k in ("id", "repo", "status", "progress", "started_at", "error")}
                    for j in _client().jobs()[:10]
                ],
                use_container_width=True,
            )
        except ApiError as exc:
            st.caption(f"unavailable: {exc.message}")

# ------------------------------------------------------------------- main --
st.title("🧠 GitMind — Codebase Archaeology")
st.write(
    "Ask natural-language questions about **why** a codebase is the way it is. Every answer is "
    "scoped to one repository and reports which evidence was retrieved, shown to the model, and cited."
)

default_repo = st.session_state.get("job_repo") or (ready_repos[0] if ready_repos else "")
if ready_repos:
    options = ready_repos + ["(enter another)"]
    choice = st.selectbox("Repository", options, index=options.index(default_repo) if default_repo in options else 0)
    repo = st.text_input("Repository (owner/name)", value=default_repo) if choice == "(enter another)" else choice
else:
    st.warning("No repository has a ready index yet. Ingest one from the sidebar.")
    repo = st.text_input("Repository (owner/name)", value=default_repo)

query = st.text_area("Your question", placeholder="Why does the auth module use JWT instead of sessions?", height=80)

c1, c2, c3 = st.columns(3)
with c1:
    mode = st.selectbox(
        "Answer mode", ["direct", "memo", "blame", "risk"],
        format_func=lambda x: {"direct": "📝 Direct Q&A", "memo": "📋 Decision Memo",
                               "blame": "👤 Blame Map", "risk": "⚠️ Risk Report"}[x],
    )
with c2:
    top_k = st.slider("Evidence chunks (top-k)", 3, 20, 12)
with c3:
    limit = st.slider("Retrieval candidates", 10, 100, 40)

with st.expander("Filters (optional)"):
    f1, f2 = st.columns(2)
    authors = f1.text_input("Authors (comma-separated)")
    modules = f2.text_input("Modules (comma-separated)")
    doc_types = f1.text_input("Document types (e.g. commit, pr, issue)")
    kind = f2.selectbox("Corpus", ["any", "history", "snapshot"])
    time_start = f1.text_input("From (ISO date, e.g. 2023-01-01)")
    time_end = f2.text_input("To (ISO date)")

if st.button("🔍 Answer", type="primary", disabled=not (query.strip() and repo.strip())):
    payload = build_query_payload(
        repo, query, mode, top_k=top_k, limit=limit, authors=authors, doc_types=doc_types,
        modules=modules, kind=kind, time_start=time_start or None, time_end=time_end or None,
    )
    with st.spinner("Retrieving and generating answer…"):
        t0 = time.time()
        try:
            data = _client().query(payload)
        except ApiError as exc:
            if exc.kind == "provider_unavailable":
                st.error(f"The language model is unavailable: {exc.message}")
            elif exc.kind == "invalid_output":
                st.error(f"The model's answer failed validation and was withheld: {exc.message}")
            elif exc.kind in ("embedding_unavailable", "index_not_ready", "index_unavailable"):
                st.error(f"Search is unavailable ({exc.kind}): {exc.message}")
            else:
                st.error(f"API error {exc.status_code}: {exc.message}")
            data = None
    if data:
        elapsed = time.time() - t0
        if data["status"] == "insufficient_evidence":
            st.warning(data["answer"])
        else:
            st.success(f"Answered in {elapsed:.1f}s · model `{data.get('model') or '?'}` · generation `{data['generation_id'][:8]}`")
            st.subheader("Answer")
            st.write(data["answer"])
        for w in data.get("warnings", []):
            st.caption(f"⚠️ {w}")
        if data.get("statistics"):
            with st.expander("Computed statistics"):
                st.json(data["statistics"])
        tabs = st.tabs([
            f"Cited ({len(data['cited'])})",
            f"Shown to model ({len(data['included'])})",
            f"Retrieved ({len(data['retrieved'])})",
        ])
        for tab, key in zip(tabs, ("cited", "included", "retrieved")):
            with tab:
                if data[key]:
                    st.dataframe(_source_rows(data[key]), use_container_width=True)
                else:
                    st.caption("none")
