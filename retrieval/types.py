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


@dataclass
class CandidateChunk:
    chunk_id: str
    score: float
    source: str
    text: str
    metadata: dict[str, Any]
