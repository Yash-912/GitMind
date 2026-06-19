"""api/main.py — FastAPI application for GitMind production deployment."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Security, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from config.settings import settings
from api.schemas import (
    QueryRequest,
    QueryResponse,
    EvidenceItem,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)

# ------------------------------------------------------------------ #
# Shared state — loaded once at startup                                #
# ------------------------------------------------------------------ #

_state: dict[str, Any] = {}
_tasks: dict[str, str] = {}          # task_id → status string


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server starts."""
    from indexing.qdrant_store import QdrantStore
    from indexing.bm25_index import BM25Index
    from indexing.fts_index import FTSIndex
    from retrieval import ChunkStore

    chunks_path = str(Path(settings.data_dir) / "chunks.jsonl")

    _state["chunk_store"] = ChunkStore(chunks_path)
    _state["qdrant"] = QdrantStore(path=settings.qdrant_path)
    _state["bm25"] = BM25Index(index_dir=settings.bm25_index_dir)
    _state["fts"] = FTSIndex(db_path=settings.db_path)
    _state["chunks_path"] = chunks_path

    yield

    # Cleanup on shutdown
    _state.get("qdrant", None) and _state["qdrant"].close()
    _state.get("fts", None) and _state["fts"].close()


# ------------------------------------------------------------------ #
# App initialisation                                                   #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="GitMind API",
    description="Codebase archaeology tool — query institutional knowledge from git history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Authentication                                                       #
# ------------------------------------------------------------------ #

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_key(api_key: str | None = Security(_api_key_header)) -> None:
    """If API_KEY is configured, enforce it on every request."""
    if not settings.api_key:
        return  # open access (dev / demo mode)
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health():
    """Returns service health and basic storage statistics."""
    qdrant_ok = False
    qdrant_count = 0
    try:
        qdrant_count = _state["qdrant"].count()
        qdrant_ok = True
    except Exception:
        pass

    bm25_chunks = _state["bm25"].size if "bm25" in _state else 0

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant_connected=qdrant_ok,
        bm25_chunks=bm25_chunks,
        details={"qdrant_vectors": qdrant_count},
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["retrieval"])
def query(req: QueryRequest, _: None = Security(_verify_key)):
    """Full pipeline: query decomposition → hybrid retrieval → graph expansion
    → reranking → LLM generation.
    """
    from retrieval import (
        HybridRetriever,
        GraphExpander,
        CrossEncoderReranker,
        ContextAssembler,
        QueryDecomposer,
        EntityResolver,
    )
    from generation import (
        DirectQAGenerator,
        DecisionMemoGenerator,
        BlameMapGenerator,
        RiskReportGenerator,
    )

    chunk_store = _state["chunk_store"]
    qdrant = _state["qdrant"]
    bm25 = _state["bm25"]
    fts = _state["fts"]

    # --- Query decomposition & entity resolution ---
    decomposer = QueryDecomposer()
    plan = decomposer.decompose(req.query)
    decomposer.close()

    resolver = EntityResolver(settings.db_path)
    module_tags = resolver.resolve(plan.entities)
    resolver.close()

    # --- Retrieval ---
    retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, fts=fts, chunk_store=chunk_store)
    candidates = retriever.retrieve(
        req.query,
        limit=req.limit,
        filters={
            "time_start": plan.time_start,
            "time_end": plan.time_end,
            "module_tags": module_tags,
        },
    )

    expander = GraphExpander(db_path=settings.db_path, chunk_store=chunk_store, hop_depth=1)
    expanded = expander.expand(candidates)
    expander.close()

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(req.query, expanded, top_k=req.top_k)

    assembler = ContextAssembler(max_tokens=3000)
    context = assembler.assemble(reranked)

    # --- Generation ---
    mode = req.mode
    if mode == "direct":
        gen = DirectQAGenerator()
        result = gen.generate(req.query, context)
        answer = result.answer
        model = result.model
    elif mode == "memo":
        gen = DecisionMemoGenerator()
        result = gen.generate(req.query, context)
        answer = result.raw_text
        model = result.model
    elif mode == "blame":
        gen = BlameMapGenerator()
        result = gen.generate(req.query, context)
        answer = result.raw_text
        model = result.model
    elif mode == "risk":
        gen = RiskReportGenerator()
        result = gen.generate(req.query, context)
        answer = result.raw_text
        model = result.model
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

    gen.close()

    # --- Build evidence list ---
    evidence = [
        EvidenceItem(
            chunk_id=c.chunk_id,
            score=round(c.score, 4),
            source=c.source,
            doc_type=c.metadata.get("doc_type", ""),
            doc_id=c.metadata.get("doc_id", ""),
            author=c.metadata.get("author", ""),
            timestamp=c.metadata.get("timestamp", ""),
            snippet=c.text[:200],
        )
        for c in reranked
    ]

    retriever.close()

    return QueryResponse(
        query=req.query,
        mode=mode,
        answer=answer,
        model=model,
        evidence=evidence,
    )


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["operations"])
def ingest(req: IngestRequest, background_tasks: BackgroundTasks, _: None = Security(_verify_key)):
    """Trigger repository ingestion as a background task."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = "started"

    def _run_ingest():
        import subprocess, sys
        _tasks[task_id] = "running"
        cmd = [sys.executable, "scripts/ingest.py", "--repo-path", req.repo_path]
        if req.github_repo:
            cmd += ["--github-repo", req.github_repo]
        if req.max_commits:
            cmd += ["--max-commits", str(req.max_commits)]
        try:
            subprocess.run(cmd, check=True)
            _tasks[task_id] = "done"
        except subprocess.CalledProcessError as exc:
            _tasks[task_id] = f"error: {exc}"

    background_tasks.add_task(_run_ingest)

    return IngestResponse(
        status="started",
        message="Ingestion started in the background.",
        task_id=task_id,
    )


@app.get("/api/v1/status/{task_id}", tags=["operations"])
def task_status(task_id: str, _: None = Security(_verify_key)):
    """Poll the status of a background ingestion task."""
    status_str = _tasks.get(task_id)
    if status_str is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": status_str}
