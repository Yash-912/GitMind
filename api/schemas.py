"""api/schemas.py — Pydantic request and response models for the GitMind REST API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from generation.answers import AnswerStatus


# ------------------------------------------------------------------ #
# Query endpoint                                                       #
# ------------------------------------------------------------------ #

class Mode(str, Enum):
    DIRECT = "direct"
    MEMO = "memo"
    BLAME = "blame"
    RISK = "risk"


def _clean_list(values: list[str]) -> list[str]:
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    return list(dict.fromkeys(cleaned))


class QueryRequest(BaseModel):
    """A question about ONE repository. `repo` is mandatory: every retrieval
    branch is scoped to it, so evidence from other corpora cannot leak in."""

    repo: str = Field(..., description="Repository identity, e.g. 'owner/name'")
    query: str = Field(..., description="Natural language question about the codebase")
    mode: Mode = Field(default=Mode.DIRECT)
    time_start: datetime | None = None
    time_end: datetime | None = None
    doc_types: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    kind: Literal["history", "snapshot"] | None = None
    top_k: int = Field(default=12, ge=1, le=40, description="Number of reranked chunks to use")
    limit: int = Field(default=40, ge=5, le=100, description="Initial retrieval candidate count")

    @field_validator("repo", "query")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("query")
    @classmethod
    def _min_len(cls, value: str) -> str:
        if len(value) < 3:
            raise ValueError("query must be at least 3 characters")
        return value

    @field_validator("doc_types", "authors", "modules")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        return _clean_list(value)

    @model_validator(mode="after")
    def _time_order(self) -> "QueryRequest":
        if self.time_start and self.time_end:
            a, b = self.time_start, self.time_end
            # Mixed naive/aware values compare as UTC (same rule as Scope).
            if (a.tzinfo is None) != (b.tzinfo is None):
                a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
            if a > b:
                raise ValueError("time_start is after time_end")
        return self


class SourceRef(BaseModel):
    """Immutable locator for one chunk of evidence."""

    chunk_id: str
    repo: str = ""
    doc_type: str = ""
    doc_id: str = ""
    timestamp: str = ""
    author: str = ""
    source_url: str = ""
    commit_sha: str = ""
    path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    score: float | None = None
    source: str = ""          # retrieval branch (dense / bm25 / fts / graph)
    snippet: str = ""


class QueryResponse(BaseModel):
    status: AnswerStatus
    query: str
    repo: str
    mode: Mode
    generation_id: str = ""
    answer: str = ""
    data: dict[str, Any] | None = None
    model: str = ""
    provider: str = ""
    error: str = ""
    # Three distinct lists: found, shown to the model, and validated citations.
    retrieved: list[SourceRef] = Field(default_factory=list)
    included: list[SourceRef] = Field(default_factory=list)
    cited: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reranker_available: bool | None = None
    statistics: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    status: str
    error: str
    detail: str = ""


# ------------------------------------------------------------------ #
# Ingest + jobs                                                        #
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
        ge=0,
        description="Limit commits to ingest (0 or omitted = all)",
    )


class IngestResponse(BaseModel):
    status: str          # "queued"
    message: str
    task_id: str | None = None   # == job_id (kept for older clients)
    job_id: str | None = None
    repo: str | None = None


class JobResponse(BaseModel):
    id: str
    repo: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    progress: str = ""
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str = ""


# ------------------------------------------------------------------ #
# Liveness / readiness                                                 #
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    """Liveness only: the process is up. Says nothing about the corpus."""

    status: Literal["ok"] = "ok"


class RepoReadiness(BaseModel):
    repo: str
    ready: bool
    generation_id: str = ""
    commit: str = ""
    chunk_count: int = 0
    embedding: dict[str, Any] = Field(default_factory=dict)
    qdrant_reachable: bool | None = None
    generation_providers: list[str] = Field(default_factory=list)
    reranker_available: bool | None = None
    degraded: list[str] = Field(default_factory=list)


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    repos: list[RepoReadiness] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
