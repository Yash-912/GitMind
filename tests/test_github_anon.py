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
