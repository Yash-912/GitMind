from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    query: str
    entities: list[str] = field(default_factory=list)
    time_start: str | None = None
    time_end: str | None = None
    intent: str = "general"
    sub_queries: list[str] = field(default_factory=list)
    # Typed entities (F05). Only `modules` may become a module filter; `people`
    # become an author filter only when `authorship` is True; `technologies`
    # never filter.
    modules: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    authorship: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateChunk:
    chunk_id: str
    score: float          # retrieval (RRF / graph) score; never overwritten by the reranker
    source: str
    text: str
    metadata: dict[str, Any]
    rerank_score: float | None = None  # cross-encoder score; None when not reranked
