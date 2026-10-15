"""multi_schema_parser.py — Per-document-type structured extraction from raw SQLite payloads."""
from __future__ import annotations

import re
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
    parent_shas: list[str] = field(default_factory=list)
    committed_at: datetime | None = None
    diff_status: str = "complete"


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
    linked_issue_numbers: list[int] = field(default_factory=list)   # local repo only
    external_issue_refs: list[str] = field(default_factory=list)    # "owner/repo#N"


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


@dataclass
class ParsedCICD:
    run_id: int
    name: str
    status: str
    conclusion: str | None
    created_at: datetime | None
    updated_at: datetime | None
    html_url: str
    event: str
    branch: str
    actor: str
    body_clean: str


@dataclass
class ParsedPRGraphQL:
    number: int
    title: str
    body_clean: str
    author: str
    created_at: datetime | None
    merged_at: datetime | None
    closing_issues: list[str]


# One repo-aware reference pattern: full GitHub URLs, ``owner/repo#N`` and bare ``#N``.
# A bare ``#N`` must not be glued to a word/path/entity (``a/b#7`` is never local).
_REF_RE = re.compile(
    r"https?://github\.com/(?P<urepo>[\w.-]+/[\w.-]+)/(?:issues|pull)/(?P<unum>\d+)"
    r"|(?<![\w/&.-])(?:(?P<repo>[\w.-]+/[\w.-]+))?#(?P<num>\d+)(?!\w)"
)
_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`|<!--.*?-->", re.DOTALL)


def extract_issue_refs(text: str, local_repo: str = "") -> tuple[list[int], list[str]]:
    """Return (local issue numbers, foreign ``owner/repo#N`` refs) found in ``text``.

    Code blocks, inline code and HTML comments are ignored. References carrying
    a repository (``owner/repo#N`` or a GitHub URL) are local only when that
    repository equals ``local_repo`` (case-insensitive); otherwise they keep
    their namespace and are never reported as local.
    """
    local: set[int] = set()
    foreign: dict[str, None] = {}
    for m in _REF_RE.finditer(_CODE_RE.sub(" ", text or "")):
        repo = m.group("urepo") or m.group("repo")
        num = int(m.group("unum") or m.group("num"))
        if repo and repo.lower() != (local_repo or "").lower():
            foreign[f"{repo}#{num}"] = None
        else:
            local.add(num)
    return sorted(local), list(foreign)


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
        # Keep every line after the subject (a blank separator is stripped, line 2 is not lost)
        body = "\n".join(lines[1:]).strip()

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
            parent_shas=payload.get("parent_shas", []),
            committed_at=_parse_dt(payload.get("committed_at")),
            diff_status=payload.get("diff_status", "complete"),
        )

    def parse_pr(self, payload: dict[str, Any]) -> ParsedPR:
        body_raw: str = payload.get("body", "") or ""
        body_clean = self._cleaner.clean_markdown(body_raw)

        # Repo-aware #NNN extraction from title + body (foreign refs keep their namespace)
        local_refs, foreign_refs = extract_issue_refs(
            (payload.get("title", "") or "") + "\n" + body_raw,
            payload.get("repo", "") or "",
        )

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
            linked_issue_numbers=local_refs,
            external_issue_refs=foreign_refs,
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

    def parse_cicd(self, payload: dict[str, Any]) -> ParsedCICD:
        summary = (
            f"Run: {payload.get('name', '')}\n"
            f"Status: {payload.get('status', '')}\n"
            f"Conclusion: {payload.get('conclusion', '')}\n"
            f"Event: {payload.get('event', '')}\n"
            f"Branch: {payload.get('branch', '')}\n"
            f"Actor: {payload.get('actor', '')}\n"
            f"URL: {payload.get('html_url', '')}"
        )
        return ParsedCICD(
            run_id=int(payload.get("run_id", 0)),
            name=payload.get("name", ""),
            status=payload.get("status", ""),
            conclusion=payload.get("conclusion"),
            created_at=_parse_dt(payload.get("created_at")),
            updated_at=_parse_dt(payload.get("updated_at")),
            html_url=payload.get("html_url", ""),
            event=payload.get("event", ""),
            branch=payload.get("branch", ""),
            actor=payload.get("actor", ""),
            body_clean=summary,
        )

    def parse_pr_graphql(self, payload: dict[str, Any]) -> ParsedPRGraphQL:
        body_raw: str = payload.get("body", "") or ""
        body_clean = self._cleaner.clean_markdown(body_raw)
        return ParsedPRGraphQL(
            number=int(payload.get("number", 0)),
            title=payload.get("title", "") or "",
            body_clean=body_clean,
            author=payload.get("author", "") or "",
            created_at=_parse_dt(payload.get("created_at")),
            merged_at=_parse_dt(payload.get("merged_at")),
            closing_issues=payload.get("closing_issues", []) or [],
        )
