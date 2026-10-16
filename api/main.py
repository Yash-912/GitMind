"""api/main.py — FastAPI application for GitMind production deployment.

Resources (per-repository active generation + retrieval service, shared LLM
client, job store) are created once in the lifespan. Endpoints are plain
``def`` so FastAPI runs them in its threadpool, keeping the event loop free.
"""
from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from api.answering import (
    DeadlineLLM,
    build_scope,
    coverage_complete,
    generate_structured_answer,
    source_ref,
    source_refs,
)
from api.jobs import JobRunner, JobStore, QueueFullError, default_steps
from api.resources import RepoNotReady, ResourceRegistry, discover_repos, probe_qdrant
from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    JobResponse,
    Mode,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RepoReadiness,
)
from config.settings import settings
from embedding.models import EmbeddingError
from generation.answers import AnswerStatus
from retrieval.scope import ScopeError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_HTTP_FOR_STATUS = {
    AnswerStatus.ANSWERED: 200,
    AnswerStatus.INSUFFICIENT_EVIDENCE: 200,
    AnswerStatus.PROVIDER_UNAVAILABLE: 503,
    AnswerStatus.INVALID_OUTPUT: 502,
}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _error(code: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"status": kind, "error": message})


# ------------------------------------------------------------------ #
# Application factory                                                  #
# ------------------------------------------------------------------ #

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_key(api_key: str | None = Security(_api_key_header)) -> None:
    """If API_KEY is configured, enforce it. This gates access to the whole API;
    it is not per-user authorization."""
    if not settings.api_key:
        return  # open access (dev / demo mode)
    if not api_key or not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def _check_repo_allowed(repo: str) -> JSONResponse | None:
    allowed = settings.allowed_repos
    if allowed and repo not in allowed:
        return _error(403, "repo_not_allowed", f"repository {repo!r} is not in the allowed set")
    return None


def create_app(
    *,
    service_factory: Callable[[Any], Any] | None = None,
    llm_factory: Callable[[], Any] | None = None,
    steps: Callable[[str, dict], Any] | None = None,
    jobs_db: str | Path | None = None,
    coverage_fn: Callable[[Any], bool] | None = None,
) -> FastAPI:
    if Path(settings.generations_dir).resolve() != (
        Path(settings.data_dir) / "generations"
    ).resolve():
        # Phase 2 always writes under DATA_DIR/generations; a different root
        # would publish generations the API never reads.
        raise ValueError(
            "GENERATIONS_DIR must equal DATA_DIR/generations: scripts/run_phase2.py "
            "writes there and the API reads from settings.generations_dir"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry_kwargs = {"service_factory": service_factory} if service_factory else {}
        registry = ResourceRegistry(settings.generations_dir, **registry_kwargs)
        if llm_factory is not None:
            llm = llm_factory()
        else:
            from generation.llm_client import LLMClient

            llm = LLMClient()
        store = JobStore(jobs_db or settings.jobs_db_path)
        store.mark_interrupted()
        runner = JobRunner(
            store,
            steps=steps or default_steps,
            queue_size=settings.job_queue_size,
            timeout_s=settings.job_timeout_s,
            on_success=registry.refresh,
        )
        app.state.registry, app.state.llm = registry, llm
        app.state.jobs, app.state.runner = store, runner
        app.state.coverage_fn = coverage_fn or coverage_complete
        try:
            repos = list(settings.allowed_repos) or discover_repos(settings.generations_dir)
            app.state.load_failures = registry.load_all(repos)
            yield
        finally:
            try:
                runner.shutdown()
            finally:
                try:
                    registry.close()
                finally:
                    try:
                        llm.close()
                    finally:
                        store.close()

    app = FastAPI(
        title="GitMind API",
        description="Codebase archaeology tool — query institutional knowledge from git history.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- liveness / readiness --------------------------------------- #

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Liveness only: the process is up. Use /ready for corpus readiness."""
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["operations"])
    def ready(request: Request, _: None = Security(_verify_key)):
        registry: ResourceRegistry = request.app.state.registry
        llm = request.app.state.llm
        repos = list(settings.allowed_repos) or sorted(
            set(registry.loaded_repos()) | set(discover_repos(settings.generations_dir))
        )
        report: list[RepoReadiness] = []
        providers = list(getattr(llm, "providers", []))
        for repo in repos:
            item = RepoReadiness(repo=repo, ready=False, generation_providers=providers)
            entry = registry.current(repo)
            if entry is None:
                err = registry.errors.get(repo) or request.app.state.load_failures.get(repo)
                item.degraded.append(f"load_failed: {err}" if err else "no_active_generation")
                report.append(item)
                continue
            gen = entry.generation
            item.generation_id, item.commit = gen.generation_id, gen.commit
            item.chunk_count, item.embedding = gen.chunk_count, dict(gen.embedding)
            item.qdrant_reachable = probe_qdrant(entry.service)
            item.reranker_available = bool(getattr(entry.service, "reranker_available", False))
            item.degraded.extend(str(w) for w in getattr(entry.service, "startup_warnings", []))
            if item.chunk_count <= 0:
                item.degraded.append("empty_corpus")
            if item.qdrant_reachable is False:
                item.degraded.append("qdrant_unreachable")
            elif item.qdrant_reachable is None:
                item.degraded.append("qdrant_status_unknown")
            if item.reranker_available is False:
                item.degraded.append("reranker_unavailable")
            if "gemini" not in providers:
                item.degraded.append("gemini_not_configured")
            item.ready = item.chunk_count > 0 and item.qdrant_reachable is not False
            report.append(item)
        overall = bool(report) and any(r.ready for r in report)
        body = ReadyResponse(
            status="ready" if overall else "not_ready",
            repos=report,
            degraded=[f"{r.repo}: {d}" for r in report for d in r.degraded],
        )
        return JSONResponse(
            status_code=200 if overall else 503, content=body.model_dump(mode="json")
        )

    # ---- query ------------------------------------------------------- #

    @app.post("/api/v1/query", response_model=QueryResponse, tags=["retrieval"])
    def query(req: QueryRequest, request: Request, _: None = Security(_verify_key)):
        """Scoped retrieval + validated generation. Outcomes are explicit:
        answered / insufficient_evidence (200), invalid_output (502),
        provider_unavailable or embedding outage (503)."""
        denied = _check_repo_allowed(req.repo)
        if denied is not None:
            return denied
        try:
            scope = build_scope(req)
        except ScopeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        deadline_at = time.monotonic() + settings.query_timeout_s
        registry: ResourceRegistry = request.app.state.registry
        llm = DeadlineLLM(request.app.state.llm, deadline_at)
        try:
            lease = registry.lease(req.repo)
            entry = lease.__enter__()
        except RepoNotReady as exc:
            return _error(503, "index_not_ready", str(exc))
        except Exception as exc:
            return _error(503, "index_unavailable", f"{type(exc).__name__}: {exc}")
        try:
            try:
                outcome = entry.service.retrieve(
                    req.query, scope, top_k=req.top_k, limit=req.limit
                )
            except EmbeddingError as exc:
                return _error(503, "embedding_unavailable", f"{type(exc).__name__}: {exc}")
            if time.monotonic() > deadline_at:
                return _error(504, "deadline_exceeded", "query exceeded its time budget during retrieval")

            result = generate_structured_answer(
                req.mode, req.query, scope, outcome, entry, llm, request.app.state.coverage_fn
            )
            candidates = list(outcome.candidates)
            payload = QueryResponse(
                status=result.status,
                query=req.query,
                repo=req.repo,
                mode=req.mode,
                generation_id=entry.generation_id,
                answer=result.text,
                data=result.data,
                model=result.model,
                provider=result.provider,
                error=result.error,
                retrieved=[source_ref(c) for c in candidates],
                included=source_refs(candidates, result.included_ids),
                cited=source_refs(candidates, result.cited_ids),
                warnings=list(getattr(outcome, "warnings", []) or []),
                reranker_available=getattr(outcome, "reranker_available", None),
                statistics=(result.data or {}).get("statistics")
                if req.mode in (Mode.BLAME, Mode.RISK) else None,
            )
            return JSONResponse(
                status_code=_HTTP_FOR_STATUS[result.status],
                content=payload.model_dump(mode="json"),
            )
        finally:
            lease.__exit__(None, None, None)

    # ---- ingestion jobs --------------------------------------------- #

    def _resolve_repo(req: IngestRequest) -> str:
        from ingestion.repository import normalize_github_repo, repository_identity

        if req.github_repo:
            return normalize_github_repo(req.github_repo)
        from git import Repo

        path = Path(req.repo_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with Repo(path) as local:
            return repository_identity(local)

    def _job(row: dict) -> JobResponse:
        return JobResponse(
            **{k: row[k] for k in (
                "id", "repo", "status", "progress", "created_at",
                "started_at", "finished_at", "error",
            )}
        )

    @app.post("/api/v1/ingest", response_model=IngestResponse, status_code=202, tags=["operations"])
    def ingest(req: IngestRequest, request: Request, _: None = Security(_verify_key)):
        """Queue a durable ingestion job (ingest -> parse/chunk -> index generation)."""
        try:
            repo = _resolve_repo(req)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"cannot resolve repository: {exc}")
        denied = _check_repo_allowed(repo)
        if denied is not None:
            return denied
        try:
            job = request.app.state.runner.submit(repo, req.model_dump())
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        return IngestResponse(
            status="queued",
            message="Ingestion queued.",
            task_id=job["id"],
            job_id=job["id"],
            repo=repo,
        )

    @app.get("/api/v1/jobs", response_model=list[JobResponse], tags=["operations"])
    def list_jobs(request: Request, repo: str | None = None, limit: int = 50,
                  _: None = Security(_verify_key)):
        return [_job(r) for r in request.app.state.jobs.list(repo=repo, limit=max(1, min(limit, 200)))]

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["operations"])
    def get_job(job_id: str, request: Request, _: None = Security(_verify_key)):
        row = request.app.state.jobs.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job(row)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse, tags=["operations"])
    def cancel_job(job_id: str, request: Request, _: None = Security(_verify_key)):
        row = request.app.state.runner.cancel(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job(row)

    @app.get("/api/v1/status/{task_id}", tags=["operations"])
    def task_status(task_id: str, request: Request, _: None = Security(_verify_key)):
        """Backward-compatible alias for GET /api/v1/jobs/{id}."""
        row = request.app.state.jobs.get(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task_id": task_id, **_job(row).model_dump()}

    return app


app = create_app()
