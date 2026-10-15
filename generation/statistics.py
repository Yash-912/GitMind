"""statistics.py — Ownership and risk figures computed from chunk records.

Numbers in blame/risk reports come from here, not from an LLM reading a
handful of retrieved chunks. Every result states its denominator, date range
and the caveats that bound what it can claim. The LLM only narrates.

Input is the full list of chunk record dicts (``{"text", "metadata"}``) for the
scope, e.g. everything in ChunkStore for a repository, never a top-k sample.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Iterable

from retrieval.scope import Scope, to_utc

_FIX_WORDS = re.compile(r"\b(fix(es|ed)?|bug|hotfix|revert|regression|patch)\b", re.I)
_MAX_FILES_FOR_COCHANGE = 30  # skip bulk commits (renames, vendor drops)
_EPOCH = to_utc("1970-01-01")


def _under(path: str, prefix: str) -> bool:
    if not prefix:
        return True
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _scope_match_ignoring_modules(scope: Scope, meta: dict) -> bool:
    # A path prefix defines the module for statistics; module-tag filters would
    # wrongly exclude commits whose chunks carry no tags.
    relaxed = Scope(
        repo=scope.repo,
        time_start=scope.time_start,
        time_end=scope.time_end,
        authors=scope.authors,
        kind=scope.kind,
        commit=scope.commit,
    )
    return relaxed.matches(meta)


def _commits(records: Iterable[dict], path_prefix: str, scope: Scope | None) -> dict[str, dict]:
    """Unique commits (by sha) in scope touching path_prefix."""
    out: dict[str, dict] = {}
    for r in records:
        m = r.get("metadata", {})
        if m.get("doc_type") != "commit":
            continue
        if scope is not None and not _scope_match_ignoring_modules(scope, m):
            continue
        files = [f for f in m.get("file_paths", []) if _under(f, path_prefix)]
        if path_prefix and not files:
            continue
        sha = m.get("doc_id", "")
        if sha and sha not in out:
            out[sha] = {
                "sha": sha,
                "author": m.get("author", "") or "unknown",
                "timestamp": to_utc(m.get("timestamp")),
                "files": files,
                "all_files": m.get("file_paths", []),
                "subject": (r.get("text", "").splitlines() or [""])[0],
            }
    return out


def _span(commits: Iterable[dict]) -> tuple[str, str]:
    stamps = sorted(c["timestamp"] for c in commits if c["timestamp"])
    if not stamps:
        return "", ""
    return stamps[0].date().isoformat(), stamps[-1].date().isoformat()


def _diff_churn(records: Iterable[dict], shas: set[str], path_prefix: str) -> dict[str, list[int]]:
    """file -> [lines_added, lines_removed] from diff chunks of the given commits."""
    churn: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        m = r.get("metadata", {})
        if m.get("doc_type") != "diff" or m.get("doc_id") not in shas:
            continue
        for path in m.get("file_paths", []):
            if not _under(path, path_prefix):
                continue
            for line in r.get("text", "").splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    churn[path][0] += 1
                elif line.startswith("-") and not line.startswith("---"):
                    churn[path][1] += 1
    return churn


@dataclass
class OwnershipStats:
    path: str
    total_commits: int
    first_commit: str
    last_commit: str
    authors: list[dict[str, Any]] = field(default_factory=list)   # author, commits, share
    timeline: list[dict[str, Any]] = field(default_factory=list)  # date, author, sha, subject
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskStats:
    path: str
    total_commits: int
    first_commit: str
    last_commit: str
    distinct_authors: int
    bus_factor: int
    fix_commits: int
    fix_ratio: float
    hotspots: list[dict[str, Any]] = field(default_factory=list)   # file, commits, added, removed
    co_change: list[dict[str, Any]] = field(default_factory=list)  # file_a, file_b, commits
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _caveats(commits: dict[str, dict], coverage_complete: bool, extra: list[str]) -> list[str]:
    notes = list(extra)
    if not coverage_complete:
        notes.append(
            "Source coverage is incomplete (partial or limited collection); "
            "counts are lower bounds for the collected history only."
        )
    if not commits:
        notes.append("No commits matched the path and scope.")
    return notes


def compute_ownership(
    records: Iterable[dict],
    path_prefix: str = "",
    scope: Scope | None = None,
    *,
    coverage_complete: bool = False,
    timeline_limit: int = 50,
) -> OwnershipStats:
    records = list(records)
    commits = _commits(records, path_prefix, scope)
    counts = Counter(c["author"] for c in commits.values())
    total = len(commits)
    first, last = _span(commits.values())
    ordered = sorted(commits.values(), key=lambda c: c["timestamp"] or _EPOCH)
    timeline = [
        {
            "date": c["timestamp"].date().isoformat() if c["timestamp"] else "",
            "author": c["author"],
            "sha": c["sha"],
            "subject": c["subject"],
        }
        for c in ordered[-timeline_limit:]
    ]
    extra = []
    if total > timeline_limit:
        extra.append(f"Timeline shows the latest {timeline_limit} of {total} commits.")
    extra.append("Ownership is commit authorship, not current line-level blame.")
    return OwnershipStats(
        path=path_prefix or "(repository)",
        total_commits=total,
        first_commit=first,
        last_commit=last,
        authors=[
            {"author": a, "commits": n, "share": round(n / total, 3)}
            for a, n in counts.most_common()
        ],
        timeline=timeline,
        caveats=_caveats(commits, coverage_complete, extra),
    )


def compute_risk(
    records: Iterable[dict],
    path_prefix: str = "",
    scope: Scope | None = None,
    *,
    coverage_complete: bool = False,
    top_n: int = 10,
) -> RiskStats:
    records = list(records)
    commits = _commits(records, path_prefix, scope)
    total = len(commits)
    first, last = _span(commits.values())

    counts = Counter(c["author"] for c in commits.values())
    bus, covered = 0, 0
    for _, n in counts.most_common():
        bus += 1
        covered += n
        if total and covered / total >= 0.5:
            break

    fixes = sum(1 for c in commits.values() if _FIX_WORDS.search(c["subject"]))

    file_commits: Counter = Counter()
    for c in commits.values():
        file_commits.update(set(c["files"]))
    churn = _diff_churn(records, set(commits), path_prefix)
    hotspots = [
        {
            "file": f,
            "commits": n,
            "lines_added": churn[f][0] if f in churn else None,
            "lines_removed": churn[f][1] if f in churn else None,
        }
        for f, n in file_commits.most_common(top_n)
    ]

    pairs: Counter = Counter()
    for c in commits.values():
        files = sorted(set(c["all_files"]))
        if 1 < len(files) <= _MAX_FILES_FOR_COCHANGE:
            for a, b in combinations(files, 2):
                if _under(a, path_prefix) or _under(b, path_prefix):
                    pairs[(a, b)] += 1
    co_change = [
        {"file_a": a, "file_b": b, "commits": n}
        for (a, b), n in pairs.most_common(top_n)
        if n > 1
    ]

    extra = [
        "Fix ratio is a keyword heuristic on commit subjects "
        "(fix/bug/hotfix/revert/regression/patch), not a defect count.",
        f"Co-change ignores commits touching more than {_MAX_FILES_FOR_COCHANGE} files.",
    ]
    if hotspots and all(h["lines_added"] is None for h in hotspots):
        extra.append("Line churn unavailable: no diff chunks were collected for these commits.")
    return RiskStats(
        path=path_prefix or "(repository)",
        total_commits=total,
        first_commit=first,
        last_commit=last,
        distinct_authors=len(counts),
        bus_factor=bus if total else 0,
        fix_commits=fixes,
        fix_ratio=round(fixes / total, 3) if total else 0.0,
        hotspots=hotspots,
        co_change=co_change,
        caveats=_caveats(commits, coverage_complete, extra),
    )


def render_stats(stats: OwnershipStats | RiskStats) -> str:
    """Plain-text block placed in the prompt as ground truth for the LLM to narrate."""
    d = stats.to_dict()
    lines = [
        f"Path: {d['path']}",
        f"Commits analysed: {d['total_commits']} ({d['first_commit']} to {d['last_commit']})",
    ]
    if isinstance(stats, OwnershipStats):
        for a in d["authors"]:
            lines.append(f"- {a['author']}: {a['commits']} commits ({a['share']:.0%})")
        lines.append("Timeline:")
        for t in d["timeline"]:
            lines.append(f"  {t['date']} {t['author']} {t['sha'][:8]} {t['subject']}")
    else:
        lines.append(
            f"Distinct authors: {d['distinct_authors']}; bus factor: {d['bus_factor']}; "
            f"fix-keyword commits: {d['fix_commits']} ({d['fix_ratio']:.0%})"
        )
        lines.append("Hotspots (file, commits, +added/-removed):")
        for h in d["hotspots"]:
            churn = (
                f"+{h['lines_added']}/-{h['lines_removed']}"
                if h["lines_added"] is not None
                else "churn n/a"
            )
            lines.append(f"  {h['file']}: {h['commits']} commits, {churn}")
        if d["co_change"]:
            lines.append("Co-changed files:")
            for p in d["co_change"]:
                lines.append(f"  {p['file_a']} + {p['file_b']}: {p['commits']} commits")
    lines.append("Caveats:")
    lines.extend(f"  - {c}" for c in d["caveats"])
    return "\n".join(lines)
