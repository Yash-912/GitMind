"""scope.py — One validated query scope applied to every retrieval branch.

A Scope always names a repository. Filters apply identically to dense, BM25,
FTS and graph expansion; `matches` is the authoritative final check.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

VALID_KINDS = {"history", "snapshot"}


class ScopeError(ValueError):
    """Raised when a scope is missing or inconsistent."""


def to_utc(value: Any) -> datetime | None:
    """Normalise a datetime or ISO string to timezone-aware UTC (naive => UTC)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ScopeError(f"invalid timestamp: {value!r}") from exc
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Scope:
    repo: str
    time_start: datetime | None = None
    time_end: datetime | None = None
    modules: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    kind: str | None = None          # "history" | "snapshot" | None (both)
    commit: str = ""                 # pin snapshot results to one commit

    def __post_init__(self) -> None:
        if not self.repo or not self.repo.strip():
            raise ScopeError("a repository scope is required")
        if self.kind is not None and self.kind not in VALID_KINDS:
            raise ScopeError(f"kind must be one of {sorted(VALID_KINDS)}")
        start, end = to_utc(self.time_start), to_utc(self.time_end)
        if start and end and start > end:
            raise ScopeError("time_start is after time_end")
        object.__setattr__(self, "time_start", start)
        object.__setattr__(self, "time_end", end)

    @classmethod
    def build(
        cls,
        repo: str,
        *,
        time_start: Any = None,
        time_end: Any = None,
        modules: Iterable[str] = (),
        authors: Iterable[str] = (),
        doc_types: Iterable[str] = (),
        kind: str | None = None,
        commit: str = "",
    ) -> "Scope":
        return cls(
            repo=repo,
            time_start=time_start,
            time_end=time_end,
            modules=tuple(modules),
            authors=tuple(authors),
            doc_types=tuple(doc_types),
            kind=kind,
            commit=commit,
        )

    def matches(self, meta: dict[str, Any]) -> bool:
        """Authoritative check against a chunk's metadata dict."""
        if meta.get("repo", "") != self.repo:
            return False
        if self.doc_types and meta.get("doc_type") not in self.doc_types:
            return False
        if self.authors and meta.get("author") not in self.authors:
            return False
        if self.modules and not set(self.modules) & set(meta.get("module_tags") or []):
            return False
        if self.kind and meta.get("scope", "history") != self.kind:
            return False
        if self.commit and meta.get("commit_sha", "") != self.commit:
            return False
        if self.time_start or self.time_end:
            ts = to_utc(meta.get("timestamp"))
            if ts is None:
                return False
            if self.time_start and ts < self.time_start:
                return False
            if self.time_end and ts > self.time_end:
                return False
        return True

    def filter_ids(self, records: Iterable[dict[str, Any]]) -> list[str]:
        """chunk_ids of full chunk dicts that satisfy the scope."""
        return [
            r["metadata"]["chunk_id"]
            for r in records
            if self.matches(r.get("metadata", {}))
        ]
