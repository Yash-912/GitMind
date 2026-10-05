from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Iterable

from github import Github
from github.GithubException import GithubException
import httpx


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


@dataclass(frozen=True)
class ReleaseRecord:
    tag: str
    name: str
    body: str
    created_at: str | None
    published_at: str | None


@dataclass(frozen=True)
class WorkflowRunRecord:
    run_id: int
    name: str
    status: str
    conclusion: str | None
    created_at: str
    updated_at: str
    html_url: str
    event: str
    branch: str
    actor: str


@dataclass(frozen=True)
class GraphQLPRRecord:
    number: int
    title: str
    body: str
    author: str
    created_at: str | None
    merged_at: str | None
    closing_issues: list[str]
    closing_issues_complete: bool = True


def _dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _utc(dt: datetime) -> datetime:
    return (
        dt.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None
        else dt.astimezone(timezone.utc)
    )


def _cutoff(value: str | None) -> datetime | None:
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00"))) if value else None


class GitHubAPICollector:
    def __init__(self, token: str, repo_full_name: str) -> None:
        self.client = Github(token, per_page=100)
        self.repo = self.client.get_repo(repo_full_name)
        self.token = token
        self.repo_full_name = repo_full_name
        if "/" in repo_full_name:
            self.owner, self.repo_name = repo_full_name.split("/", 1)
        else:
            self.owner, self.repo_name = "", repo_full_name

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
        self, state: str = "all", log_every: int = 50, updated_since: str | None = None
    ) -> Iterable[PullRequestRecord]:
        self._throttle()
        pull_iter = self.repo.get_pulls(state=state, sort="updated", direction="asc")
        cutoff = _cutoff(updated_since)
        for idx, pr in enumerate(self._iter_with_throttle(pull_iter), start=1):
            if log_every and idx % log_every == 0:
                print(f"[github] pulled {idx} PRs...")
            if idx % 50 == 0:
                self._throttle()
            # Replay timestamp ties so an interruption cannot skip other PRs.
            if cutoff and _utc(pr.updated_at) < cutoff:
                continue
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

    def iter_issues(
        self, state: str = "all", log_every: int = 50, updated_since: str | None = None
    ) -> Iterable[IssueRecord]:
        self._throttle()
        issue_iter = self.repo.get_issues(state=state, sort="updated", direction="asc")
        cutoff = _cutoff(updated_since)
        for idx, issue in enumerate(self._iter_with_throttle(issue_iter), start=1):
            if log_every and idx % log_every == 0:
                print(f"[github] pulled {idx} issues...")
            if idx % 50 == 0:
                self._throttle()
            if issue.pull_request is not None:
                continue
            if cutoff and _utc(issue.updated_at) < cutoff:
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

    def collect_issues(
        self, state: str = "all", log_every: int = 50
    ) -> list[IssueRecord]:
        return list(self.iter_issues(state=state, log_every=log_every))

    def collect_releases(self, log_every: int = 20) -> list[ReleaseRecord]:
        releases = []
        for idx, rel in enumerate(
            self._iter_with_throttle(self.repo.get_releases()), start=1
        ):
            if log_every and idx % log_every == 0:
                print(f"[github] pulled {idx} releases...")
            releases.append(
                ReleaseRecord(
                    tag=rel.tag_name or "",
                    name=rel.title or "",
                    body=rel.body or "",
                    created_at=_dt(rel.created_at),
                    published_at=_dt(rel.published_at),
                )
            )
        return releases

    def collect_workflow_runs(
        self, per_page: int = 100, limit: int | None = None
    ) -> list[WorkflowRunRecord]:
        if not 1 <= per_page <= 100 or (limit is not None and limit < 1):
            raise ValueError("Invalid workflow page size or limit")
        if not self.owner:
            return []
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/actions/runs"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
        }
        runs: list[WorkflowRunRecord] = []
        page = 1
        with httpx.Client(timeout=60.0) as client:
            while limit is None or len(runs) < limit:
                resp = client.get(
                    url, headers=headers, params={"per_page": per_page, "page": page}
                )
                resp.raise_for_status()
                data = resp.json()
                for run in data.get("workflow_runs", []):
                    runs.append(
                        WorkflowRunRecord(
                            run_id=int(run.get("id", 0)),
                            name=run.get("name", ""),
                            status=run.get("status", ""),
                            conclusion=run.get("conclusion"),
                            created_at=run.get("created_at", ""),
                            updated_at=run.get("updated_at", ""),
                            html_url=run.get("html_url", ""),
                            event=run.get("event", ""),
                            branch=run.get("head_branch", ""),
                            actor=(run.get("actor") or {}).get("login", ""),
                        )
                    )
                    if limit is not None and len(runs) >= limit:
                        break
                if len(data.get("workflow_runs", [])) < per_page:
                    break
                page += 1
        return runs

    def collect_prs_graphql(self, limit: int | None = None) -> list[GraphQLPRRecord]:
        if limit is not None and limit < 1:
            raise ValueError("GraphQL PR limit must be positive")
        if not self.owner:
            return []
        query = """
        query($owner: String!, $name: String!, $pageSize: Int!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            pullRequests(first: $pageSize, after: $cursor, states: MERGED, orderBy: {field: UPDATED_AT, direction: ASC}) {
              pageInfo { hasNextPage endCursor }
              nodes {
                number
                title
                bodyText
                createdAt
                mergedAt
                author { login }
                closingIssuesReferences(first: 100) {
                  pageInfo { hasNextPage }
                  nodes { number title }
                }
              }
            }
          }
        }
        """
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
        }
        records: list[GraphQLPRRecord] = []
        cursor = None
        with httpx.Client(timeout=60.0) as client:
            while limit is None or len(records) < limit:
                page_size = min(100, limit - len(records)) if limit is not None else 100
                payload = {
                    "query": query,
                    "variables": {
                        "owner": self.owner,
                        "name": self.repo_name,
                        "pageSize": page_size,
                        "cursor": cursor,
                    },
                }
                resp = client.post(
                    "https://api.github.com/graphql", headers=headers, json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("errors"):
                    raise RuntimeError(
                        "GitHub GraphQL returned errors; collection is incomplete"
                    )
                repository = (data.get("data") or {}).get("repository")
                if repository is None:
                    raise RuntimeError(
                        "GitHub GraphQL did not return the requested repository"
                    )
                connection = repository["pullRequests"]
                for pr in connection["nodes"]:
                    issues = pr.get("closingIssuesReferences", {}).get("nodes", [])
                    closing = [f"#{i.get('number')}: {i.get('title')}" for i in issues]
                    records.append(
                        GraphQLPRRecord(
                            number=int(pr.get("number", 0)),
                            title=pr.get("title", ""),
                            body=pr.get("bodyText", ""),
                            author=(pr.get("author") or {}).get("login", ""),
                            created_at=pr.get("createdAt"),
                            merged_at=pr.get("mergedAt"),
                            closing_issues=closing,
                            closing_issues_complete=not pr.get(
                                "closingIssuesReferences", {}
                            )
                            .get("pageInfo", {})
                            .get("hasNextPage", False),
                        )
                    )
                page_info = connection["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                next_cursor = page_info.get("endCursor")
                if not next_cursor or next_cursor == cursor:
                    raise RuntimeError("GitHub GraphQL pagination did not advance")
                cursor = next_cursor
        return records

    def close(self) -> None:
        self.client.close()
