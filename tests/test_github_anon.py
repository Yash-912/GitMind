"""The collector must build an unauthenticated client when no token is given."""
from ingestion import github_collector


class _FakeGithub:
    calls = []

    def __init__(self, *args, **kwargs):
        _FakeGithub.calls.append((args, kwargs))

    def get_repo(self, name):
        return object()


def test_no_token_builds_unauthenticated_client(monkeypatch):
    _FakeGithub.calls = []
    monkeypatch.setattr(github_collector, "Github", _FakeGithub)
    github_collector.GitHubAPICollector("", "octocat/Hello-World")
    assert _FakeGithub.calls == [((), {"per_page": 100})]


def test_token_is_passed_through(monkeypatch):
    _FakeGithub.calls = []
    monkeypatch.setattr(github_collector, "Github", _FakeGithub)
    github_collector.GitHubAPICollector("tok", "octocat/Hello-World")
    assert _FakeGithub.calls == [(("tok",), {"per_page": 100})]


def test_unauthenticated_rate_limit_fails_fast(monkeypatch):
    import datetime as dt
    import pytest

    class _Rate:
        remaining = 0
        reset = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=50)

    class _Limits:
        core = _Rate()

    class _Client:
        def get_rate_limit(self):
            return _Limits()

    collector = github_collector.GitHubAPICollector.__new__(github_collector.GitHubAPICollector)
    collector.token = ""
    collector.client = _Client()
    slept = []
    monkeypatch.setattr(github_collector.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        collector._throttle()
    assert slept == []
