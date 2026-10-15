"""Computed ownership/risk statistics and their use in blame/risk generators."""
from __future__ import annotations

from generation.blame_map import BlameMapGenerator
from generation.risk_report import RiskReportGenerator
from generation.llm_client import LLMResponse
from generation.statistics import compute_ownership, compute_risk, render_stats
from retrieval.scope import Scope


def _commit(sha, author, ts, files, subject="change", repo="o/r"):
    return {
        "text": f"{subject}\n\nFiles changed: {', '.join(files)}",
        "metadata": {
            "chunk_id": f"commit-{sha}", "doc_type": "commit", "doc_id": sha,
            "author": author, "timestamp": ts, "file_paths": files, "repo": repo,
        },
    }


def _diff(sha, path, lines, repo="o/r"):
    return {
        "text": "File: x\n@@ -1 +1 @@\n" + "\n".join(lines),
        "metadata": {
            "chunk_id": f"diff-{sha}-{path}", "doc_type": "diff", "doc_id": sha,
            "author": "x", "timestamp": "2022-01-01T00:00:00", "file_paths": [path],
            "repo": repo,
        },
    }


RECORDS = [
    _commit("a1", "alice", "2021-01-10T00:00:00", ["auth/login.py", "auth/token.py"], "add login"),
    _commit("a2", "alice", "2021-03-01T00:00:00", ["auth/login.py"], "fix login bug"),
    _commit("b1", "bob", "2021-06-01T00:00:00", ["auth/login.py", "auth/token.py"], "refactor"),
    _commit("c1", "carol", "2021-07-01T00:00:00", ["billing/pay.py"], "add billing"),
    _commit("z1", "mallory", "2021-08-01T00:00:00", ["auth/login.py"], "other repo", repo="x/y"),
    _diff("a1", "auth/login.py", ["+a", "+b", "-c", " ctx"]),
    _diff("a2", "auth/login.py", ["+d", "-e", "-f"]),
]


def test_ownership_counts_and_denominator():
    s = compute_ownership(RECORDS, "auth", Scope.build("o/r"))
    assert s.total_commits == 3
    assert {a["author"]: a["commits"] for a in s.authors} == {"alice": 2, "bob": 1}
    assert s.authors[0]["share"] == round(2 / 3, 3)
    assert (s.first_commit, s.last_commit) == ("2021-01-10", "2021-06-01")


def test_ownership_excludes_other_repos_and_paths():
    s = compute_ownership(RECORDS, "auth", Scope.build("o/r"))
    assert "mallory" not in {a["author"] for a in s.authors}
    assert all("carol" != a["author"] for a in s.authors)


def test_ownership_time_scope_applies():
    scope = Scope.build("o/r", time_start="2021-02-01", time_end="2021-12-31")
    s = compute_ownership(RECORDS, "auth", scope)
    assert s.total_commits == 2 and {a["author"] for a in s.authors} == {"alice", "bob"}


def test_timeline_is_chronological_and_complete_flag_caveat():
    s = compute_ownership(RECORDS, "auth", Scope.build("o/r"))
    dates = [t["date"] for t in s.timeline]
    assert dates == sorted(dates)
    assert any("incomplete" in c for c in s.caveats)
    done = compute_ownership(RECORDS, "auth", Scope.build("o/r"), coverage_complete=True)
    assert not any("incomplete" in c for c in done.caveats)


def test_duplicate_commit_pieces_counted_once():
    dup = RECORDS + [_commit("a1", "alice", "2021-01-10T00:00:00", ["auth/login.py"], "add login")]
    assert compute_ownership(dup, "auth", Scope.build("o/r")).total_commits == 3


def test_empty_scope_reports_no_commits_not_zero_division():
    s = compute_ownership(RECORDS, "nope", Scope.build("o/r"))
    assert s.total_commits == 0 and s.authors == []
    assert any("No commits" in c for c in s.caveats)
    r = compute_risk(RECORDS, "nope", Scope.build("o/r"))
    assert r.total_commits == 0 and r.bus_factor == 0 and r.fix_ratio == 0.0


def test_risk_hotspots_churn_and_ratio():
    r = compute_risk(RECORDS, "auth", Scope.build("o/r"))
    top = r.hotspots[0]
    assert top["file"] == "auth/login.py" and top["commits"] == 3
    assert (top["lines_added"], top["lines_removed"]) == (3, 3)
    assert r.hotspots[1]["lines_added"] is None  # no diff chunks for token.py
    assert r.fix_commits == 1 and r.fix_ratio == round(1 / 3, 3)
    assert r.distinct_authors == 2 and r.bus_factor == 1  # alice alone covers >= 50%


def test_risk_co_change_pairs():
    r = compute_risk(RECORDS, "auth", Scope.build("o/r"))
    assert r.co_change == [{"file_a": "auth/login.py", "file_b": "auth/token.py", "commits": 2}]


def test_render_includes_numbers_and_caveats():
    text = render_stats(compute_risk(RECORDS, "auth", Scope.build("o/r")))
    assert "auth/login.py: 3 commits, +3/-3" in text
    assert "bus factor: 1" in text and "Caveats:" in text


class _LLM:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt):
        self.prompt = prompt
        return LLMResponse(text="narrative", model="fake")

    def close(self):
        pass


def test_blame_generator_injects_computed_stats():
    llm = _LLM()
    res = BlameMapGenerator(llm).generate("auth", "ctx", records=RECORDS, scope=Scope.build("o/r"))
    assert "alice: 2 commits" in llm.prompt and res.stats.total_commits == 3


def test_risk_generator_injects_computed_stats():
    llm = _LLM()
    res = RiskReportGenerator(llm).generate("auth", "ctx", records=RECORDS, scope=Scope.build("o/r"))
    assert "bus factor: 1" in llm.prompt and res.stats.fix_commits == 1


def test_generators_without_records_keep_old_behaviour():
    llm = _LLM()
    res = BlameMapGenerator(llm).generate("auth", "ctx")
    assert res.stats is None and "Computed statistics" not in llm.prompt
