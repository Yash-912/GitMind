from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from git import Repo


@dataclass(frozen=True)
class CommitRecord:
    sha: str
    author_name: str
    author_email: str
    authored_at: str
    message: str
    file_paths: list[str]
    diff_text: str
    file_changes: list[dict]
    stats: dict


@dataclass(frozen=True)
class FileChange:
    path: str
    old_path: str | None
    new_path: str | None
    change_type: str
    additions: int
    deletions: int


def _isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class GitCollector:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.repo = Repo(repo_path)

    def iter_commits(
        self,
        since_sha: str | None = None,
        max_count: int | None = None,
    ) -> Iterable[CommitRecord]:
        count = 0
        for commit in self.repo.iter_commits():
            if since_sha and commit.hexsha == since_sha:
                break
            if max_count is not None and count >= max_count:
                break

            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent, create_patch=True)
            diff_text = "".join(d.diff.decode("utf-8", errors="replace") for d in diffs)
            file_paths = [d.b_path or d.a_path or "" for d in diffs]
            stats = commit.stats
            file_stats = stats.files

            file_changes: list[FileChange] = []
            for d in diffs:
                old_path = d.a_path
                new_path = d.b_path
                path = new_path or old_path or ""
                stat_key = path if path in file_stats else old_path or new_path or ""
                stat = file_stats.get(stat_key, {})
                file_changes.append(
                    FileChange(
                        path=path,
                        old_path=old_path,
                        new_path=new_path,
                        change_type=d.change_type or "",
                        additions=int(stat.get("insertions", 0)),
                        deletions=int(stat.get("deletions", 0)),
                    )
                )

            record = CommitRecord(
                sha=commit.hexsha,
                author_name=commit.author.name,
                author_email=commit.author.email,
                authored_at=_isoformat(commit.authored_datetime),
                message=commit.message.strip(),
                file_paths=[p for p in file_paths if p],
                diff_text=diff_text,
                file_changes=[fc.__dict__ for fc in file_changes],
                stats={
                    "files": int(stats.total.get("files", 0)),
                    "insertions": int(stats.total.get("insertions", 0)),
                    "deletions": int(stats.total.get("deletions", 0)),
                },
            )
            yield record
            count += 1

    def collect_commits(
        self,
        since_sha: str | None = None,
        max_count: int | None = None,
    ) -> list[CommitRecord]:
        return list(self.iter_commits(since_sha=since_sha, max_count=max_count))
