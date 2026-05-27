from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


def _dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


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

    def collect_releases(self, log_every: int = 20) -> list[ReleaseRecord]:
        releases = []
        for idx, rel in enumerate(self._iter_with_throttle(self.repo.get_releases()), start=1):
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

    def collect_workflow_runs(self, per_page: int = 50, limit: int = 200) -> list[WorkflowRunRecord]:
        if not self.owner:
            return []
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/actions/runs"
        headers = {"Authorization": f"token {self.token}", "Accept": "application/vnd.github+json"}
        runs: list[WorkflowRunRecord] = []
        page = 1
        with httpx.Client(timeout=60.0) as client:
            while len(runs) < limit:
                resp = client.get(url, headers=headers, params={"per_page": per_page, "page": page})
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
                    if len(runs) >= limit:
                        break
                if len(data.get("workflow_runs", [])) < per_page:
                    break
                page += 1
        return runs

    def collect_prs_graphql(self, limit: int = 50) -> list[GraphQLPRRecord]:
        if not self.owner:
            return []
        query = """
        query($owner: String!, $name: String!, $limit: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequests(last: $limit, states: MERGED, orderBy: {field: UPDATED_AT, direction: DESC}) {
              nodes {
                number
                title
                bodyText
                createdAt
                mergedAt
                author { login }
                closingIssuesReferences(first: 5) {
                  nodes { number title }
                }
              }
            }
          }
        }
        """
        headers = {"Authorization": f"token {self.token}", "Accept": "application/vnd.github+json"}
        payload = {
            "query": query,
            "variables": {"owner": self.owner, "name": self.repo_name, "limit": limit},
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.github.com/graphql", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        nodes = (
            data.get("data", {})
            .get("repository", {})
            .get("pullRequests", {})
            .get("nodes", [])
        )
        records: list[GraphQLPRRecord] = []
        for pr in nodes:
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
                )
            )
        return records
