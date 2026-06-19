"""api/schemas.py — Pydantic request and response models for the GitMind REST API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Query endpoint                                                       #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language question about the codebase")
    mode: str = Field(
        default="direct",
        description="Answer mode: 'direct' | 'memo' | 'blame' | 'risk'",
    )
    top_k: int = Field(default=12, ge=1, le=40, description="Number of reranked chunks to use")
    limit: int = Field(default=40, ge=5, le=100, description="Initial retrieval candidate count")


class EvidenceItem(BaseModel):
    chunk_id: str
    score: float
    source: str
    doc_type: str = ""
    doc_id: str = ""
    author: str = ""
    timestamp: str = ""
    snippet: str = ""


class QueryResponse(BaseModel):
    query: str
    mode: str
    answer: str
    model: str
    evidence: list[EvidenceItem] = []


# ------------------------------------------------------------------ #
# Ingest endpoint                                                      #
# ------------------------------------------------------------------ #

class IngestRequest(BaseModel):
    repo_path: str = Field(
        default=".",
        description="Local path to the git repository to ingest",
    )
    github_repo: str | None = Field(
        default=None,
        description="GitHub repo slug, e.g. 'tiangolo/fastapi'",
    )
    max_commits: int | None = Field(
        default=None,
        description="Limit commits to ingest (useful for testing)",
    )


class IngestResponse(BaseModel):
    status: str          # "started" | "error"
    message: str
    task_id: str | None = None


# ------------------------------------------------------------------ #
# Health endpoint                                                      #
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    qdrant_connected: bool
    bm25_chunks: int
    details: dict[str, Any] = {}
