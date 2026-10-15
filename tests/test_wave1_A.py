"""tests/test_wave1_A.py — Regression tests for audit F01, F02, F21, F22, F23."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import (
    ChangelogChunker,
    ChunkMetadata,
    CommitChunker,
    DiffChunker,
    IssueChunker,
    LogChunker,
    PRChunker,
)
from chunking.chunk_metadata import stable_chunk_id
from parsing.multi_schema_parser import (
    MultiSchemaParser,
    ParsedCICD,
    ParsedIssue,
    ParsedPR,
    ParsedRelease,
    extract_issue_refs,
)

TS = datetime(2024, 1, 2, 3, 4, 5)
parser = MultiSchemaParser()


def _commit(message="subject\n\nbody", diff_text="", sha="a" * 40):
    return parser.parse_commit(
        {
            "sha": sha,
            "message": message,
            "author_name": "dev",
            "authored_at": "2024-01-02T03:04:05Z",
            "file_paths": ["src/a.py"],
            "diff_text": diff_text,
            "stats": {"insertions": 1, "deletions": 0, "files": 1},
        }
    )


def _diff(n_lines, new_start=10, line_len=20):
    body = "\n".join(f"+{i:04d}{'x' * line_len}" for i in range(n_lines))
    return (
        "--- a/src/a.py\n+++ b/src/a.py\n"
        f"@@ -{new_start},0 +{new_start},{n_lines} @@ def f():\n{body}\n"
    )


def _issue(body="body", comments=None):
    return ParsedIssue(
        number=7, title="T", body_clean=body, state="open", author="u",
        created_at=TS, closed_at=None, labels=["bug"], comments=comments or [],
    )


def _pr(body="body", review_comments=None):
    return ParsedPR(
        number=3, title="T", body_clean=body, state="open", author="u",
        created_at=TS, merged_at=None, labels=[], review_comments=review_comments or [],
    )


def _ids(chunks):
    return [c.metadata.chunk_id for c in chunks]


def _all_chunks(repo="o/r"):
    out = []
    out += CommitChunker().chunk(_commit(), repo=repo)
    out += DiffChunker().chunk(_commit(diff_text=_diff(5)), repo=repo)
    out += PRChunker().chunk(_pr(review_comments=["lgtm"]), repo=repo)
    out += IssueChunker().chunk(_issue(comments=["a", "b"]), repo=repo)
    out += ChangelogChunker().chunk(
        ParsedRelease(tag="v1", timestamp=TS, body_clean="## 1.0.0 - 2020-01-01\n- x"), repo=repo
    )
    cicd = ParsedCICD(
        run_id=1, name="ci", status="done", conclusion="success", created_at=TS,
        updated_at=TS, html_url="", event="push", branch="main", actor="a", body_clean="ok",
    )
    out += LogChunker("cicd").chunk(cicd, repo=repo)
    return out


# ---------------------------------------------------------------- F01

class TestDeterministicIds:
    def test_same_input_same_ids_across_runs(self):
        assert _ids(_all_chunks()) == _ids(_all_chunks())

    def test_all_chunkers_assign_nonempty_unique_ids(self):
        ids = _ids(_all_chunks())
        assert all(ids) and len(ids) == len(set(ids))

    def test_repo_changes_identity(self):
        assert _ids(_all_chunks("o/r")) != _ids(_all_chunks("o/other"))

    def test_text_changes_identity(self):
        a = CommitChunker().chunk(_commit("s\n\nv1"))[0].metadata.chunk_id
        b = CommitChunker().chunk(_commit("s\n\nv2"))[0].metadata.chunk_id
        assert a != b

    def test_identical_text_chunks_get_distinct_ids(self):
        chunks = IssueChunker(window_size=1, overlap=0).chunk(
            _issue(comments=["dup", "dup"])
        )
        ids = _ids(chunks)
        assert len(ids) == len(set(ids))

    def test_stable_chunk_id_is_pure(self):
        args = ("r", "commit", "abc", ["f"], "text", 0)
        assert stable_chunk_id(*args) == stable_chunk_id(*args)
        assert stable_chunk_id(*args) != stable_chunk_id("r", "commit", "abc", ["f"], "text", 1)

    def test_default_metadata_has_no_random_id(self):
        assert ChunkMetadata().chunk_id == ChunkMetadata().chunk_id == ""


# ---------------------------------------------------------------- F02

class TestChangelogSplitIds:
    def _release(self):
        section = "\n".join(f"- change number {i} with some padding text" for i in range(80))
        body = f"## 2.0.0 - 2022-01-01\n{section}\n\n## 1.0.0 - 2021-01-01\n- small"
        return ParsedRelease(tag="r", timestamp=TS, body_clean=body), body

    def test_split_pieces_have_unique_ids_and_offsets(self):
        release, body = self._release()
        chunks = ChangelogChunker(max_tokens=100).chunk(release, repo="o/r")
        big = [c for c in chunks if c.metadata.doc_id == "2.0.0"]
        assert len(big) > 1
        ids = _ids(chunks)
        assert len(ids) == len(set(ids))
        for c in big:
            m = c.metadata
            assert m.char_start is not None and m.char_end > m.char_start
            assert body[m.char_start:m.char_end] in c.text

    def test_split_pieces_cover_whole_section(self):
        release, body = self._release()
        chunks = ChangelogChunker(max_tokens=100).chunk(release)
        big = sorted(
            (c for c in chunks if c.metadata.doc_id == "2.0.0"),
            key=lambda c: c.metadata.char_start,
        )
        start = body.index("- change number 0")
        end = body.index("\n\n## 1.0.0")
        assert big[0].metadata.char_start == start
        assert big[-1].metadata.char_end == end
        for prev, nxt in zip(big, big[1:]):
            assert nxt.metadata.char_start <= prev.metadata.char_end  # no gap


# ---------------------------------------------------------------- F22

def _covered(spans, n):
    covered = [False] * n
    for s, e in spans:
        for i in range(s, e):
            covered[i] = True
    return all(covered)


class TestNoSilentTruncation:
    def test_split_by_chars_has_no_redundant_tail(self):
        ch = CommitChunker(max_tokens=10)
        text = "x" * ch.max_chars
        assert ch._split_by_chars(text) == [text]
        long = "".join(f"{i:05d}" for i in range(ch.max_chars))[: ch.max_chars * 3]
        pieces = ch._split_by_chars(long, overlap_chars=5)
        assert all(len(p) <= ch.max_chars for p in pieces)
        # lossless: stitching pieces minus overlap rebuilds the text
        rebuilt = pieces[0] + "".join(p[5:] for p in pieces[1:])
        assert rebuilt == long
        # no piece is contained in the previous one
        for a, b in zip(pieces, pieces[1:]):
            assert b not in a

    def test_commit_long_message_is_split_not_truncated(self):
        ch = CommitChunker(max_tokens=20)
        message = "subject\n\n" + " ".join(f"word{i}" for i in range(200)) + " TAILMARK"
        chunks = ch.chunk(_commit(message))
        assert len(chunks) > 1
        assert any("TAILMARK" in c.text for c in chunks)
        assert all(len(c.text) <= ch.max_chars for c in chunks)
        assert all(c.metadata.total_chunks == len(chunks) for c in chunks)
        assert all(c.metadata.commit_sha == "a" * 40 for c in chunks)

    def test_issue_long_body_split(self):
        ch = IssueChunker(max_tokens=50)
        body = "".join(f"line{i}\n" for i in range(300)) + "ENDMARK"
        chunks = ch.chunk(_issue(body=body))
        assert len(chunks) > 1
        assert any("ENDMARK" in c.text for c in chunks)
        assert all(len(c.text) <= ch.max_chars for c in chunks)
        assert _covered([(c.metadata.char_start, c.metadata.char_end) for c in chunks], len(body))
        assert all(c.text.startswith("[Issue #7]") for c in chunks)

    def test_issue_long_comment_window_split(self):
        ch = IssueChunker(max_tokens=50)
        comments = ["first " * 80, "second " * 80 + "LASTMARK"]
        chunks = ch.chunk(_issue(body="b", comments=comments))
        assert any("LASTMARK" in c.text for c in chunks)
        assert all(len(c.text) <= ch.max_chars for c in chunks)

    def test_issue_windows_have_no_redundant_trailing_window(self):
        chunks = IssueChunker().chunk(_issue(comments=["c1", "c2", "c3"]))
        assert len(chunks) == 2  # body + one window of 3

    def test_pr_long_body_and_review_comment_split(self):
        ch = PRChunker(max_tokens=60)
        body = "# A\n" + "alpha " * 200 + "\n# B\n" + "beta " * 200 + "ENDBODY"
        review = "review " * 200 + "ENDREVIEW"
        chunks = ch.chunk(_pr(body=body, review_comments=[review]))
        assert any("ENDBODY" in c.text for c in chunks)
        assert any("ENDREVIEW" in c.text for c in chunks)
        assert all(len(c.text) <= ch.max_chars for c in chunks)
        ids = _ids(chunks)
        assert len(ids) == len(set(ids))
        body_spans = [
            (c.metadata.char_start, c.metadata.char_end)
            for c in chunks
            if "review_comment" not in c.metadata.entity_tags
        ]
        covered = {i for s, e in body_spans for i in range(s, e)}
        stripped = body.strip()
        assert all(i in covered for i, ch_ in enumerate(stripped) if not ch_.isspace())

    def test_release_without_sections_split(self):
        ch = ChangelogChunker(max_tokens=50)
        body = "note " * 300 + "ENDNOTE"
        chunks = ch.chunk(ParsedRelease(tag="v9", timestamp=TS, body_clean=body))
        assert len(chunks) > 1
        assert any("ENDNOTE" in c.text for c in chunks)
        assert all(len(c.text) <= ch.max_chars for c in chunks)

    def test_diff_split_pieces_have_real_line_ranges(self):
        ch = DiffChunker(max_tokens=60)
        n = 60
        chunks = ch.chunk(_commit(diff_text=_diff(n, new_start=10)))
        assert len(chunks) > 1
        for c in chunks:
            m = c.metadata
            assert m.source_path == "src/a.py"
            assert m.commit_sha == "a" * 40
            assert m.scope == "history"
            assert 10 <= m.line_start <= m.line_end <= 10 + n - 1
            # the first and last line of the range appear in the text
            assert f"+{m.line_start - 10:04d}" in c.text
            assert f"+{m.line_end - 10:04d}" in c.text
        assert chunks[0].metadata.line_start == 10
        assert chunks[-1].metadata.line_end == 10 + n - 1
        starts = [c.metadata.line_start for c in chunks]
        assert starts == sorted(starts)

    def test_diff_overlong_single_line_not_dropped(self):
        ch = DiffChunker(max_tokens=50)
        chunks = ch.chunk(_commit(diff_text=_diff(1, line_len=2000)))
        assert len(chunks) > 1
        assert all(c.metadata.line_start == c.metadata.line_end == 10 for c in chunks)
        total_x = sum(c.text.count("x") for c in chunks)
        assert total_x >= 2000

    def test_metadata_to_dict_has_provenance_fields(self):
        d = ChunkMetadata().to_dict()
        for key in ("commit_sha", "source_path", "line_start", "line_end",
                    "char_start", "char_end", "scope"):
            assert key in d
        assert d["scope"] == "history"


# ---------------------------------------------------------------- F21

class TestCommitBody:
    def test_line_two_kept_without_blank_separator(self):
        c = parser.parse_commit({"message": "subject\nsecond\nthird"})
        assert c.message_subject == "subject"
        assert c.message_body == "second\nthird"

    def test_two_line_message_keeps_second_line(self):
        assert parser.parse_commit({"message": "subject\nsecond"}).message_body == "second"

    def test_blank_separator_still_stripped(self):
        c = parser.parse_commit({"message": "subject\n\nbody line\nmore"})
        assert c.message_body == "body line\nmore"

    def test_subject_only(self):
        assert parser.parse_commit({"message": "subject"}).message_body == ""


# ---------------------------------------------------------------- F23

class TestIssueRefs:
    def test_foreign_ref_not_local(self):
        local, foreign = extract_issue_refs("see other/repo#7 and fixes #5")
        assert local == [5]
        assert foreign == ["other/repo#7"]

    def test_urls_and_code_ignored_as_local(self):
        text = (
            "https://example.com/page#12 and https://github.com/o/r/issues/9\n"
            "`#11` and\n```\n#12\n```\n<!-- #13 -->\nreal #14"
        )
        local, foreign = extract_issue_refs(text)
        assert local == [14]
        assert foreign == ["o/r#9"]

    def test_same_repo_qualified_ref_is_local(self):
        local, foreign = extract_issue_refs("Me/Proj#7 other/x#8", local_repo="me/proj")
        assert local == [7]
        assert foreign == ["other/x#8"]

    def test_word_glued_hash_not_a_ref(self):
        assert extract_issue_refs("C#1 foo#2 a/b/c#3")[0] == []

    def test_parse_pr_separates_local_and_foreign(self):
        pr = parser.parse_pr(
            {
                "number": 1,
                "title": "Fix #5",
                "body": "Closes #6, relates to other/repo#7 and https://github.com/x/y/pull/8",
            }
        )
        assert pr.linked_issue_numbers == [5, 6]
        assert pr.external_issue_refs == ["other/repo#7", "x/y#8"]
