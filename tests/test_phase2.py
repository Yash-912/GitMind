"""tests/test_phase2.py — Unit tests for all Phase 2 components."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# =====================================================================
# TextCleaner
# =====================================================================

class TestTextCleaner:
    def setup_method(self):
        from parsing.text_cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_clean_markdown_strips_html_comments(self):
        text = "Hello <!-- hidden --> World"
        result = self.cleaner.clean_markdown(text)
        assert "hidden" not in result
        assert "Hello" in result

    def test_clean_markdown_strips_checkboxes(self):
        text = "- [ ] Task A\n- [x] Task B done"
        result = self.cleaner.clean_markdown(text)
        assert "[ ]" not in result

    def test_clean_markdown_collapses_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = self.cleaner.clean_markdown(text)
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert self.cleaner.clean_markdown("") == ""
        assert self.cleaner.clean_html("") == ""


# =====================================================================
# DiffParser
# =====================================================================

SAMPLE_DIFF = """\
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,8 @@ def login(username, password):
     user = get_user(username)
     if not user:
         raise NotFound()
+    if not verify_token(user):
+        raise Unauthorized()
     return user
"""

class TestDiffParser:
    def setup_method(self):
        from parsing.diff_parser import DiffParser
        self.parser = DiffParser()

    def test_parse_basic_diff(self):
        hunks = self.parser.parse(SAMPLE_DIFF)
        assert len(hunks) >= 1
        hunk = hunks[0]
        assert "auth.py" in hunk.file_path
        assert hunk.old_start == 10

    def test_added_lines(self):
        hunks = self.parser.parse(SAMPLE_DIFF)
        added = hunks[0].added_lines
        assert any("verify_token" in line for line in added)

    def test_empty_diff(self):
        assert self.parser.parse("") == []
        assert self.parser.parse("   ") == []

    def test_token_estimate(self):
        hunks = self.parser.parse(SAMPLE_DIFF)
        assert hunks[0].token_estimate() > 0


# =====================================================================
# CodeParser
# =====================================================================

SAMPLE_PYTHON = """\
def hello(name: str) -> str:
    return f"Hello, {name}"

class MyClass:
    def method(self):
        pass
"""

class TestCodeParser:
    def setup_method(self):
        from parsing.code_parser import CodeParser
        self.parser = CodeParser()

    def test_parse_functions(self):
        entities = self.parser.parse(SAMPLE_PYTHON, "test.py")
        names = [e.name for e in entities]
        assert "hello" in names

    def test_parse_class(self):
        entities = self.parser.parse(SAMPLE_PYTHON, "test.py")
        class_entities = [e for e in entities if e.entity_type == "class"]
        assert any(e.name == "MyClass" for e in class_entities)

    def test_empty_source(self):
        result = self.parser.parse("", "empty.py")
        assert isinstance(result, list)


# =====================================================================
# MultiSchemaParser
# =====================================================================

class TestMultiSchemaParser:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        self.parser = MultiSchemaParser()

    def _commit_payload(self):
        return {
            "sha": "abc123",
            "author_name": "Alice",
            "author_email": "alice@example.com",
            "authored_at": "2023-01-15T10:00:00+00:00",
            "message": "fix: handle auth edge case\n\nThis fixes the JWT expiry bug.",
            "file_paths": ["src/auth.py"],
            "diff_text": SAMPLE_DIFF,
            "stats": {"files": 1, "insertions": 2, "deletions": 0},
        }

    def test_parse_commit(self):
        pc = self.parser.parse_commit(self._commit_payload())
        assert pc.sha == "abc123"
        assert pc.message_subject == "fix: handle auth edge case"
        assert "JWT" in pc.message_body
        assert len(pc.hunks) >= 1

    def test_parse_pr(self):
        payload = {
            "number": 42,
            "title": "Add JWT auth (#17)",
            "body": "Fixes #17. This PR adds JWT authentication.",
            "state": "closed",
            "author": "bob",
            "created_at": "2023-02-01T12:00:00",
            "merged_at": "2023-02-05T15:00:00",
            "labels": ["enhancement"],
            "review_comments": ["LGTM"],
        }
        pr = self.parser.parse_pr(payload)
        assert pr.number == 42
        assert 17 in pr.linked_issue_numbers
        assert "JWT" in pr.body_clean

    def test_parse_issue(self):
        payload = {
            "number": 17,
            "title": "Auth broken after deploy",
            "body": "The login endpoint returns 500.",
            "state": "closed",
            "author": "carol",
            "created_at": "2023-01-20T09:00:00",
            "closed_at": "2023-02-05T15:00:00",
            "labels": ["bug"],
            "comments": ["Confirmed.", "Fixed in #42."],
        }
        issue = self.parser.parse_issue(payload)
        assert issue.number == 17
        assert len(issue.comments) == 2


# =====================================================================
# ChunkMetadata
# =====================================================================

class TestChunkMetadata:
    def test_to_dict_roundtrip(self):
        from chunking.chunk_metadata import ChunkMetadata
        meta = ChunkMetadata(
            doc_type="commit",
            doc_id="abc123",
            author="alice",
            repo="myrepo",
        )
        d = meta.to_dict()
        assert d["doc_type"] == "commit"
        assert d["doc_id"] == "abc123"
        assert "chunk_id" in d
        assert "timestamp" in d


# =====================================================================
# CommitChunker
# =====================================================================

class TestCommitChunker:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        from chunking.commit_chunker import CommitChunker
        self.parser = MultiSchemaParser()
        self.chunker = CommitChunker()

    def _make_commit(self):
        return self.parser.parse_commit({
            "sha": "deadbeef",
            "author_name": "Alice",
            "author_email": "a@b.com",
            "authored_at": "2023-03-10T08:00:00+00:00",
            "message": "feat: add rate limiter\n\nAdds Redis-based rate limiting.",
            "file_paths": ["app/limiter.py"],
            "diff_text": "",
            "stats": {"files": 1, "insertions": 50, "deletions": 0},
        })

    def test_produces_one_chunk(self):
        chunks = self.chunker.chunk(self._make_commit(), repo="myrepo")
        assert len(chunks) == 1

    def test_chunk_has_subject(self):
        chunks = self.chunker.chunk(self._make_commit())
        assert "rate limiter" in chunks[0].text

    def test_metadata_doc_type(self):
        chunks = self.chunker.chunk(self._make_commit())
        assert chunks[0].metadata.doc_type == "commit"


# =====================================================================
# DiffChunker
# =====================================================================

class TestDiffChunker:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        from chunking.diff_chunker import DiffChunker
        self.parser = MultiSchemaParser()
        self.chunker = DiffChunker()

    def _make_commit(self):
        return self.parser.parse_commit({
            "sha": "cafebabe",
            "author_name": "Bob",
            "author_email": "b@c.com",
            "authored_at": "2023-04-01T12:00:00+00:00",
            "message": "fix: patch auth",
            "file_paths": ["src/auth.py"],
            "diff_text": SAMPLE_DIFF,
            "stats": {"files": 1, "insertions": 2, "deletions": 0},
        })

    def test_produces_chunks(self):
        chunks = self.chunker.chunk(self._make_commit(), repo="myrepo")
        assert len(chunks) >= 1

    def test_file_path_in_text(self):
        chunks = self.chunker.chunk(self._make_commit())
        assert any("auth.py" in c.text for c in chunks)

    def test_metadata_doc_type(self):
        chunks = self.chunker.chunk(self._make_commit())
        assert all(c.metadata.doc_type == "diff" for c in chunks)


# =====================================================================
# PRChunker
# =====================================================================

class TestPRChunker:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        from chunking.pr_chunker import PRChunker
        self.parser = MultiSchemaParser()
        self.chunker = PRChunker()

    def test_short_pr_single_chunk(self):
        pr = self.parser.parse_pr({
            "number": 1,
            "title": "Tiny fix",
            "body": "One-liner fix.",
            "state": "closed",
            "author": "dev",
            "created_at": "2023-01-01T00:00:00",
            "merged_at": None,
            "labels": [],
            "review_comments": [],
        })
        chunks = self.chunker.chunk(pr)
        # At least one body chunk
        assert len(chunks) >= 1

    def test_review_comment_chunk(self):
        pr = self.parser.parse_pr({
            "number": 2,
            "title": "Feature",
            "body": "Does stuff.",
            "state": "open",
            "author": "dev",
            "created_at": "2023-01-01T00:00:00",
            "merged_at": None,
            "labels": [],
            "review_comments": ["Please fix the typo.", "LGTM!"],
        })
        chunks = self.chunker.chunk(pr)
        review_chunks = [c for c in chunks if "review comment" in c.text]
        assert len(review_chunks) == 2


# =====================================================================
# IssueChunker
# =====================================================================

class TestIssueChunker:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        from chunking.issue_chunker import IssueChunker
        self.parser = MultiSchemaParser()
        self.chunker = IssueChunker(window_size=3, overlap=1)

    def test_body_chunk(self):
        issue = self.parser.parse_issue({
            "number": 5,
            "title": "Bug report",
            "body": "Something broke.",
            "state": "open",
            "author": "user",
            "created_at": "2023-05-01T00:00:00",
            "closed_at": None,
            "labels": ["bug"],
            "comments": [],
        })
        chunks = self.chunker.chunk(issue)
        assert len(chunks) >= 1
        assert "Bug report" in chunks[0].text

    def test_sliding_window_comments(self):
        issue = self.parser.parse_issue({
            "number": 6,
            "title": "Long discussion",
            "body": "Issue body.",
            "state": "closed",
            "author": "user",
            "created_at": "2023-05-01T00:00:00",
            "closed_at": "2023-05-10T00:00:00",
            "labels": [],
            "comments": [f"Comment {i}" for i in range(10)],
        })
        chunks = self.chunker.chunk(issue)
        # 1 body + multiple comment windows
        assert len(chunks) > 1


# =====================================================================
# ChangelogChunker
# =====================================================================

SAMPLE_CHANGELOG = """\
## v2.0.0 — 2023-06-01

Major release with breaking changes.

## v1.9.0 — 2023-05-01

Added new features.

## v1.8.5 — 2023-04-01

Bug fixes.
"""

class TestChangelogChunker:
    def setup_method(self):
        from parsing.multi_schema_parser import MultiSchemaParser
        from chunking.changelog_chunker import ChangelogChunker
        self.parser = MultiSchemaParser()
        self.chunker = ChangelogChunker()

    def test_splits_versions(self):
        release = self.parser.parse_release({
            "tag": "CHANGELOG",
            "body": SAMPLE_CHANGELOG,
            "published_at": None,
        })
        chunks = self.chunker.chunk(release)
        assert len(chunks) >= 3
        texts = " ".join(c.text for c in chunks)
        assert "v2.0.0" in texts
        assert "v1.9.0" in texts

    def test_single_release_one_chunk(self):
        release = self.parser.parse_release({
            "tag": "v1.0.0",
            "body": "Initial release.",
            "published_at": "2022-01-01T00:00:00",
        })
        chunks = self.chunker.chunk(release)
        assert len(chunks) == 1


# =====================================================================
# EntityExtractor
# =====================================================================

class TestEntityExtractor:
    def setup_method(self):
        from entities.entity_extractor import EntityExtractor
        self.extractor = EntityExtractor()

    def test_detects_tech_keyword(self):
        entities = self.extractor.extract(
            "We switched from sessions to JWT authentication.",
            doc_id="commit_abc",
            doc_type="commit",
        )
        labels = {e.label for e in entities}
        names = {e.text.lower() for e in entities}
        assert "jwt" in names or any("JWT" in e.text for e in entities)

    def test_empty_text(self):
        result = self.extractor.extract("", doc_id="x", doc_type="commit")
        assert isinstance(result, list)


# =====================================================================
# EntityRegistry
# =====================================================================

class TestEntityRegistry:
    def setup_method(self):
        import tempfile, os
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        from entities.entity_registry import EntityRegistry
        self.registry = EntityRegistry(self._tmpfile.name)

    def teardown_method(self):
        self.registry.close()
        import os
        os.unlink(self._tmpfile.name)

    def test_register_and_get(self):
        self.registry.register("JWT", "tech", aliases=["jsonwebtoken"])
        record = self.registry.get("JWT")
        assert record is not None
        assert "jsonwebtoken" in record.aliases

    def test_resolve_exact_match(self):
        self.registry.register("JWT", "tech")
        canonical = self.registry.resolve("JWT", "tech")
        assert canonical == "JWT"

    def test_resolve_alias_match(self):
        self.registry.register("JWT", "tech", aliases=["jsonwebtoken"])
        canonical = self.registry.resolve("jsonwebtoken", "tech")
        assert canonical == "JWT"

    def test_resolve_new_entity(self):
        canonical = self.registry.resolve("NewTech", "tech")
        assert canonical == "NewTech"
        assert self.registry.get("NewTech") is not None

    def test_add_doc_reference(self):
        self.registry.register("auth", "module")
        self.registry.add_doc_reference("auth", "commit_abc123")
        record = self.registry.get("auth")
        assert "commit_abc123" in record.doc_ids


# =====================================================================
# TemporalGraph
# =====================================================================

class TestTemporalGraph:
    def setup_method(self):
        import tempfile
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        from entities.temporal_graph import TemporalGraphBuilder, TemporalGraphWalker
        self.builder = TemporalGraphBuilder(self._tmpfile.name)
        self.walker = TemporalGraphWalker(self._tmpfile.name, hop_depth=1)

    def teardown_method(self):
        self.builder.close()
        self.walker.close()
        import os
        os.unlink(self._tmpfile.name)

    def test_add_and_retrieve_edge(self):
        self.builder.add_edge("commit", "abc", "pr", "42", "merge_commit")
        neighbors = self.builder.get_neighbors("commit", "abc", direction="out")
        assert any(n["type"] == "pr" and n["id"] == "42" for n in neighbors)

    def test_bidirectional_lookup(self):
        self.builder.add_edge("pr", "42", "issue", "17", "mentions")
        out = self.builder.get_neighbors("pr", "42", direction="out")
        in_ = self.builder.get_neighbors("issue", "17", direction="in")
        assert any(n["id"] == "17" for n in out)
        assert any(n["id"] == "42" for n in in_)

    def test_edge_count(self):
        self.builder.add_edge("commit", "a", "pr", "1", "merge_commit")
        self.builder.add_edge("pr", "1", "issue", "5", "mentions")
        assert self.builder.edge_count() == 2

    def test_duplicate_edges_ignored(self):
        self.builder.add_edge("commit", "a", "pr", "1", "merge_commit")
        self.builder.add_edge("commit", "a", "pr", "1", "merge_commit")
        assert self.builder.edge_count() == 1

    def test_walker_1hop(self):
        self.builder.add_edge("commit", "x", "pr", "10", "merge_commit")
        self.builder.add_edge("pr", "10", "issue", "3", "mentions")
        neighbors = self.walker.expand("commit", "x")
        ids = {n["id"] for n in neighbors}
        assert "10" in ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
