"""multi_schema_parser.py — Per-document-type structured extraction from raw SQLite payloads."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .diff_parser import DiffParser, DiffHunk
from .text_cleaner import TextCleaner


@dataclass
class ParsedCommit:
    sha: str
    author: str
    author_email: str
    timestamp: datetime
    message: str
    message_subject: str      # first line of commit message
    message_body: str         # remaining lines
    file_paths: list[str]
    hunks: list[DiffHunk]
    stats: dict[str, int]


@dataclass
class ParsedPR:
    number: int
    title: str
    body_clean: str
    state: str
    author: str
    created_at: datetime | None
    merged_at: datetime | None
    labels: list[str]
    review_comments: list[str]
    linked_issue_numbers: list[int] = field(default_factory=list)


@dataclass
class ParsedIssue:
    number: int
    title: str
    body_clean: str
    state: str
    author: str
    created_at: datetime | None
    closed_at: datetime | None
    labels: list[str]
    comments: list[str]


@dataclass
class ParsedRelease:
    tag: str
    timestamp: datetime | None
    body_clean: str


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class MultiSchemaParser:
    """Transforms raw document payloads (as stored in SQLite) into typed dataclasses."""

    def __init__(self) -> None:
        self._diff_parser = DiffParser()
        self._cleaner = TextCleaner()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def parse_commit(self, payload: dict[str, Any]) -> ParsedCommit:
        message: str = payload.get("message", "")
        lines = message.splitlines()
        subject = lines[0].strip() if lines else ""
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""

        diff_text: str = payload.get("diff_text", "")
        hunks = self._diff_parser.parse(diff_text)

        return ParsedCommit(
            sha=payload.get("sha", ""),
            author=payload.get("author_name", ""),
            author_email=payload.get("author_email", ""),
            timestamp=_parse_dt(payload.get("authored_at")) or datetime.utcnow(),
            message=message,
            message_subject=subject,
            message_body=body,
            file_paths=payload.get("file_paths", []),
            hunks=hunks,
            stats=payload.get("stats", {}),
        )

    def parse_pr(self, payload: dict[str, Any]) -> ParsedPR:
        import re

        body_raw: str = payload.get("body", "") or ""
        body_clean = self._cleaner.clean_markdown(body_raw)

        # Extract #NNN issue references from body + title
        issue_refs = [
            int(n)
            for n in re.findall(r"#(\d+)", (payload.get("title", "") or "") + body_raw)
        ]

        return ParsedPR(
            number=int(payload.get("number", 0)),
            title=payload.get("title", "") or "",
            body_clean=body_clean,
            state=payload.get("state", ""),
            author=payload.get("author", ""),
            created_at=_parse_dt(payload.get("created_at")),
            merged_at=_parse_dt(payload.get("merged_at")),
            labels=payload.get("labels", []) or [],
            review_comments=payload.get("review_comments", []) or [],
            linked_issue_numbers=list(set(issue_refs)),
        )

    def parse_issue(self, payload: dict[str, Any]) -> ParsedIssue:
        body_raw: str = payload.get("body", "") or ""
        body_clean = self._cleaner.clean_markdown(body_raw)
        comments_raw: list[str] = payload.get("comments", []) or []
        comments_clean = [self._cleaner.clean_markdown(c) for c in comments_raw]

        return ParsedIssue(
            number=int(payload.get("number", 0)),
            title=payload.get("title", "") or "",
            body_clean=body_clean,
            state=payload.get("state", ""),
            author=payload.get("author", ""),
            created_at=_parse_dt(payload.get("created_at")),
            closed_at=_parse_dt(payload.get("closed_at")),
            labels=payload.get("labels", []) or [],
            comments=comments_clean,
        )

    def parse_release(self, payload: dict[str, Any]) -> ParsedRelease:
        body_raw: str = payload.get("body", "") or ""
        body_clean = self._cleaner.clean_markdown(body_raw)
        return ParsedRelease(
            tag=payload.get("tag", payload.get("tag_name", "")),
            timestamp=_parse_dt(payload.get("timestamp") or payload.get("published_at")),
            body_clean=body_clean,
        )
