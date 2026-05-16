from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Iterable

from github import Github
from github.GithubException import GithubException


@dataclass(frozen=True)
class PullRequestRecord:
    number: int
    title: str
    body: str
    state: str
    author: str
    created_at: str
    updated_at: str
    closed_at: str | None
    merged_at: str | None
    merge_commit_sha: str | None
    labels: list[str]
    review_comments: list[str]


@dataclass(frozen=True)
class IssueRecord:
    number: int
    title: str
    body: str
    state: str
    author: str
    created_at: str
    updated_at: str
    closed_at: str | None
    labels: list[str]
    comments: list[str]


def _dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


class GitHubAPICollector:
    def __init__(self, token: str, repo_full_name: str) -> None:
        self.client = Github(token, per_page=100)
        self.repo = self.client.get_repo(repo_full_name)

    def _throttle(self) -> None:
        rate_overview = self.client.get_rate_limit()
        rate = (
            rate_overview.core
            if hasattr(rate_overview, "core")
            else rate_overview.resources.core
        )
        if rate.remaining <= 1:
            sleep_for = max(rate.reset.timestamp() - time.time(), 0) + 1
            time.sleep(sleep_for)

    def _iter_with_throttle(self, iterator: Iterable) -> Iterable:
        try:
            for item in iterator:
                yield item
        except GithubException:
            self._throttle()
            raise

    def iter_pull_requests(
        self, state: str = "all", log_every: int = 50
    ) -> Iterable[PullRequestRecord]:
        self._throttle()
        pull_iter = self.repo.get_pulls(state=state)
        for idx, pr in enumerate(self._iter_with_throttle(pull_iter), start=1):
            if log_every and idx % log_every == 0:
                print(f"[github] pulled {idx} PRs...")
            if idx % 50 == 0:
                self._throttle()
            labels = [label.name for label in pr.labels]
            review_comments = [c.body or "" for c in pr.get_review_comments()]
            yield PullRequestRecord(
                number=pr.number,
                title=pr.title or "",
                body=pr.body or "",
                state=pr.state,
                author=pr.user.login if pr.user else "",
                created_at=_dt(pr.created_at) or "",
                updated_at=_dt(pr.updated_at) or "",
                closed_at=_dt(pr.closed_at),
                merged_at=_dt(pr.merged_at),
                merge_commit_sha=pr.merge_commit_sha,
                labels=labels,
                review_comments=review_comments,
            )

    def iter_issues(self, state: str = "all", log_every: int = 50) -> Iterable[IssueRecord]:
        self._throttle()
        issue_iter = self.repo.get_issues(state=state)
        for idx, issue in enumerate(self._iter_with_throttle(issue_iter), start=1):
            if log_every and idx % log_every == 0:
                print(f"[github] pulled {idx} issues...")
            if idx % 50 == 0:
                self._throttle()
            if issue.pull_request is not None:
                continue
            labels = [label.name for label in issue.labels]
            comments = [c.body or "" for c in issue.get_comments()]
            yield IssueRecord(
                number=issue.number,
                title=issue.title or "",
                body=issue.body or "",
                state=issue.state,
                author=issue.user.login if issue.user else "",
                created_at=_dt(issue.created_at) or "",
                updated_at=_dt(issue.updated_at) or "",
                closed_at=_dt(issue.closed_at),
                labels=labels,
                comments=comments,
            )

    def collect_pull_requests(
        self, state: str = "all", log_every: int = 50
    ) -> list[PullRequestRecord]:
        return list(self.iter_pull_requests(state=state, log_every=log_every))

    def collect_issues(self, state: str = "all", log_every: int = 50) -> list[IssueRecord]:
        return list(self.iter_issues(state=state, log_every=log_every))
