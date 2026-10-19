"""Workflow runs must follow GitHub's 301 from the repo-name URL to the numeric repo id.

httpx does not follow redirects unless asked, so the collector failed with a
301 on the first page of actions/runs and the whole ingestion run was marked
failed (found in a full run of tiangolo/fastapi).
"""
from __future__ import annotations

import httpx

from ingestion import github_collector
from ingestion.github_collector import GitHubAPICollector


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/repos/o/r/actions/runs":
        return httpx.Response(
            301,
            headers={"location": "https://api.github.com/repositories/1/actions/runs"},
        )
    if request.url.path == "/repositories/1/actions/runs":
        return httpx.Response(200, json={"workflow_runs": [{"id": 7, "name": "ci"}]})
    return httpx.Response(404)


class _Repo:
    pass


class _FakeGithub:
    def __init__(self, *args, **kwargs):
        pass

    def get_repo(self, name):
        return _Repo()


def test_listing_cap_422_ends_history_without_error(monkeypatch):
    def capped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repositories/1/actions/runs" or request.url.path == "/repos/o/r/actions/runs":
            page = int(request.url.params["page"])
            if page <= 2:
                runs = [{"id": page * 1000 + i} for i in range(100)]
                return httpx.Response(200, json={"workflow_runs": runs})
            return httpx.Response(422, json={"message": "Pagination cap"})
        return httpx.Response(404)

    monkeypatch.setattr(github_collector, "Github", _FakeGithub)
    real_client = httpx.Client

    def client_with_mock(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(capped)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(github_collector.httpx, "Client", client_with_mock)
    runs = GitHubAPICollector("tok", "o/r").collect_workflow_runs()
    assert len(runs) == 200


def test_workflow_runs_follow_repo_id_redirect(monkeypatch):
    monkeypatch.setattr(github_collector, "Github", _FakeGithub)
    real_client = httpx.Client

    def client_with_mock(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(github_collector.httpx, "Client", client_with_mock)
    runs = GitHubAPICollector("tok", "o/r").collect_workflow_runs(limit=5)
    assert [r.run_id for r in runs] == [7]
