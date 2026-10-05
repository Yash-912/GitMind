from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from git import NULL_TREE, Repo
from git.exc import GitCommandError


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
    parent_shas: list[str] = field(default_factory=list)
    committed_at: str = ""
    diff_base_sha: str | None = None
    diff_status: str = "complete"
    diff_error: str | None = None


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
    def __init__(self, repo_path: str, allow_incomplete: bool = False) -> None:
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.allow_incomplete = allow_incomplete

    @property
    def is_shallow(self) -> bool:
        return self.repo.git.rev_parse("--is-shallow-repository").strip() == "true"

    def close(self) -> None:
        self.repo.close()

    def iter_commits(
        self,
        since_sha: str | None = None,
        max_count: int | None = None,
        oldest_first: bool = False,
        exclude_shas: set[str] | None = None,
    ) -> Iterable[CommitRecord]:
        if max_count is not None and max_count < 1:
            raise ValueError("max_count must be positive, or None for all commits")
        if self.is_shallow and not self.allow_incomplete:
            raise ValueError(
                "Repository history is shallow. Fetch the complete history or explicitly allow incomplete history."
            )
        if not self.repo.head.is_valid():
            return
        revision = "HEAD"
        if since_sha:
            try:
                self.repo.commit(since_sha)
                self.repo.git.merge_base("--is-ancestor", since_sha, "HEAD")
            except (ValueError, GitCommandError) as exc:
                raise ValueError(
                    "The saved commit is not an ancestor of HEAD. Use a full collection to reset the checkpoint."
                ) from exc
            revision = f"{since_sha}..HEAD"
        count = 0
        for commit in self.repo.iter_commits(
            rev=revision, reverse=oldest_first, topo_order=True
        ):
            if exclude_shas and commit.hexsha in exclude_shas:
                continue
            if max_count is not None and count >= max_count:
                break

            parent_shas = [p.hexsha for p in commit.parents]
            diff_status, diff_error = "complete", None
            diff_text = ""
            try:
                parent = commit.parents[0] if commit.parents else None
                # Diff the old tree against the new tree, including root additions.
                diffs = parent.diff(commit) if parent else commit.diff(NULL_TREE)
                if parent:
                    diff_text = self.repo.git.diff(
                        parent.hexsha,
                        commit.hexsha,
                        "--",
                        no_ext_diff=True,
                        no_textconv=True,
                        no_color=True,
                        find_renames=True,
                    )
                else:
                    diff_text = self.repo.git.show(
                        commit.hexsha,
                        format="",
                        root=True,
                        no_ext_diff=True,
                        no_textconv=True,
                        no_color=True,
                        find_renames=True,
                    )
                if diff_text:
                    diff_text += "\n"
                stats_obj = commit.stats
                stats_total = {
                    "files": int(stats_obj.total.get("files", 0)),
                    "insertions": int(stats_obj.total.get("insertions", 0)),
                    "deletions": int(stats_obj.total.get("deletions", 0)),
                }
                file_stats = stats_obj.files
            except (GitCommandError, ValueError, OSError) as exc:
                if not self.allow_incomplete:
                    raise RuntimeError(
                        f"Cannot collect diff/stats for commit {commit.hexsha}; no successful record was stored."
                    ) from exc
                diff_status = "unavailable"
                diff_error = f"{type(exc).__name__}: diff or statistics unavailable"
                diff_text = ""
                diffs = []
                stats_total = {"files": 0, "insertions": 0, "deletions": 0}
                file_stats = {}

            file_paths = [d.b_path or d.a_path or "" for d in diffs]

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

            author_name = commit.author.name if commit.author else ""
            author_email = commit.author.email if commit.author else ""

            record = CommitRecord(
                sha=commit.hexsha,
                author_name=author_name or "",
                author_email=author_email or "",
                authored_at=_isoformat(commit.authored_datetime),
                message=commit.message.strip(),
                file_paths=[p for p in file_paths if p],
                diff_text=diff_text,
                file_changes=[fc.__dict__ for fc in file_changes],
                stats=stats_total,
                parent_shas=parent_shas,
                committed_at=_isoformat(commit.committed_datetime),
                diff_base_sha=parent_shas[0] if parent_shas else None,
                diff_status=diff_status,
                diff_error=diff_error,
            )
            yield record
            count += 1

    def collect_commits(
        self,
        since_sha: str | None = None,
        max_count: int | None = None,
        oldest_first: bool = False,
    ) -> list[CommitRecord]:
        return list(
            self.iter_commits(
                since_sha=since_sha, max_count=max_count, oldest_first=oldest_first
            )
        )
