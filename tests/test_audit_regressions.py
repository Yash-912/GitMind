"""Regression tests for PROJECT_AUDIT.md findings, with corrected expectations.

docs/audit_checks.py asserted the defects as they existed before the fixes.
These tests assert the fixed behaviour for the same 11 reproductions.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from chunking.changelog_chunker import ChangelogChunker
from chunking.commit_chunker import CommitChunker
from chunking.log_chunker import LogChunker
from config.settings import settings
from embedding.embedder import DualEmbedder
from embedding.embedding_cache import EmbeddingCache
from embedding.models import EmbeddingError, OllamaEmbeddingClient
from entities.tagging import module_tags_for_paths
from evaluation.custom_metrics.entity_consistency import EntityConsistencyMetric
from evaluation.custom_metrics.temporal_accuracy import TemporalAccuracyMetric
from generation.llm_client import LLMClient, LLMUnavailableError
from indexing.fts_index import FTSIndex
from parsing.multi_schema_parser import MultiSchemaParser
from retrieval.context_assembler import ContextAssembler
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.scope import Scope
from retrieval.types import CandidateChunk


@pytest.fixture
def parser():
    return MultiSchemaParser()


@pytest.fixture
def commit(parser):
    return parser.parse_commit({
        "sha": "abc", "message": "Subject\nCritical second line\nThird line",
        "authored_at": "2023-01-01T00:00:00+00:00", "file_paths": ["src/auth.py"],
    })


# F01: deterministic identities
def test_identical_input_gives_identical_chunk_ids(commit):
    first = CommitChunker().chunk(commit, repo="owner/repo")[0]
    second = CommitChunker().chunk(commit, repo="owner/repo")[0]
    assert first.metadata.chunk_id == second.metadata.chunk_id


# F21: no line loss in commit bodies
def test_commit_body_without_blank_separator_keeps_second_line(commit):
    first = CommitChunker().chunk(commit, repo="owner/repo")[0]
    assert "Critical second line" in first.text


# F05: module tags derived from file paths
def test_module_tags_derived_from_file_paths():
    assert module_tags_for_paths(["src/auth.py"]) == ["auth"]


# F02: split release pieces keep distinct identities
def test_split_release_pieces_have_distinct_ids(parser):
    release = parser.parse_release({
        "tag": "v1.0", "published_at": "2023-01-01T00:00:00+00:00",
        "body": "## v1.0\n" + "x" * 6000,
    })
    chunks = ChangelogChunker().chunk(release)
    assert len(chunks) > 1
    assert len({c.metadata.chunk_id for c in chunks}) == len(chunks)


# F01/F09: re-indexing the same chunk does not accumulate FTS records
def test_fts_replaces_identical_chunk_instead_of_duplicating(commit):
    chunk = CommitChunker().chunk(commit, repo="owner/repo")[0]
    fts = FTSIndex(":memory:")
    try:
        record = {"text": chunk.text, "metadata": chunk.metadata.to_dict()}
        fts.add_chunks([record])
        fts.add_chunks([record])
        assert fts.count() == 1
    finally:
        fts.close()


# F06: a total embedding outage raises, never returns or caches zero vectors
def test_total_embedding_outage_raises_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    client = OllamaEmbeddingClient()
    client._client.close()
    client._client = SimpleNamespace(
        post=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated outage")),
        close=lambda: None,
    )
    client._gemini = None
    cache = EmbeddingCache(cache_dir=str(tmp_path / "cache"))
    embedder = DualEmbedder(client=client, cache=cache)
    try:
        with pytest.raises(EmbeddingError):
            embedder.embed_chunks(
                [{"text": "content", "metadata": {"chunk_id": "a", "doc_type": "commit"}}],
                show_progress=False,
            )
        assert cache.get("content", embedder.prose_model) is None
    finally:
        embedder.close()


# F19: total generation outage is an error, not an empty successful answer
def test_total_generation_outage_raises(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    llm = LLMClient()
    llm._http.close()
    llm._http = SimpleNamespace(
        post=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated outage")),
        close=lambda: None,
    )
    try:
        with pytest.raises(LLMUnavailableError):
            llm.generate("test")
    finally:
        llm.close()


# F15: an oversized first candidate does not empty the context
def test_oversized_first_candidate_does_not_empty_context():
    assembler = ContextAssembler(max_tokens=20)
    assembler._tokenizer = None
    out = assembler.assemble_with_report([
        CandidateChunk("large", 1.0, "dense", "x" * 500, {}),
        CandidateChunk("small", 0.5, "dense", "short evidence", {}),
    ])
    # The context is not empty, the budget holds, and every candidate is accounted for.
    assert out.text
    assert out.included_ids or out.truncated_ids
    assert assembler._count_tokens(out.text) <= 20
    accounted = set(out.included_ids) | set(out.skipped_ids)
    assert accounted == {"large", "small"}


# F28: unsupported years and entities lower the custom scores
def test_unsupported_year_lowers_temporal_score():
    score = TemporalAccuracyMetric().score(
        ["It happened in 2023 and 1999"], [["2023-01-01"]]
    )
    assert score is not None and score < 1.0


def test_unsupported_entity_lowers_consistency_score():
    score = EntityConsistencyMetric().score(
        ["JWT uses Redis and Kafka"], [[["JWT"]]]
    )
    assert score is not None and score < 1.0


# F14: GraphQL closing references appear in the chunk text under the canonical node id
def test_graphql_closing_references_reach_chunk_text(parser):
    gql = parser.parse_pr_graphql({
        "number": 42, "body": "Change", "closing_issues": ["#7: Motivation"],
    })
    chunk = LogChunker("pr").chunk(gql)[0]
    assert "#7" in chunk.text
    assert chunk.metadata.graph_node_id == "pr_42"


# F04: a candidate excluded by the scope never reaches the results
class _Store:
    def __init__(self, records):
        self._records = {r["metadata"]["chunk_id"]: r for r in records}

    def get(self, chunk_id):
        return self._records.get(chunk_id)

    def records(self):
        return list(self._records.values())


def test_scope_excludes_candidate_from_every_branch():
    excluded = {"text": "Excluded issue", "metadata": {
        "chunk_id": "excluded", "doc_type": "issue", "author": "bob", "repo": "owner/repo",
        "timestamp": "1999-01-01T00:00:00+00:00", "module_tags": ["payments"], "scope": "history",
    }}
    retriever = HybridRetriever(
        qdrant=SimpleNamespace(search_semantic=lambda **kwargs: []),
        bm25=SimpleNamespace(search=lambda *a, **k: [SimpleNamespace(chunk_id="excluded")]),
        fts=SimpleNamespace(search=lambda *a, **k: [SimpleNamespace(chunk_id="excluded")]),
        chunk_store=_Store([excluded]),
        embedder=SimpleNamespace(embed=lambda text: [1.0]),
    )
    scope = Scope.build(
        "owner/repo", doc_types=["commit"], authors=["alice"],
        modules=["auth"], time_start="2023-01-01",
    )
    results = retriever.retrieve("test", scope=scope)
    assert "excluded" not in [r.chunk_id for r in results]


def test_legacy_filters_path_excludes_candidate_failing_every_filter():
    """Unscoped filters= must be enforced after fusion on every branch, not only dense."""
    excluded = {"text": "Excluded issue", "metadata": {
        "chunk_id": "excluded", "doc_type": "issue", "author": "bob", "repo": "owner/repo",
        "timestamp": "1999-01-01T00:00:00+00:00", "module_tags": ["payments"], "scope": "history",
    }}
    retriever = HybridRetriever(
        qdrant=SimpleNamespace(search_semantic=lambda **kwargs: []),
        bm25=SimpleNamespace(search=lambda *a, **k: [SimpleNamespace(chunk_id="excluded")]),
        fts=SimpleNamespace(search=lambda *a, **k: [SimpleNamespace(chunk_id="excluded")]),
        chunk_store=_Store([excluded]),
        embedder=SimpleNamespace(embed=lambda text: [1.0]),
        use_code_search=False,
    )
    results = retriever.retrieve("test", filters={
        "doc_types": ["commit"], "author": "alice", "module_tags": ["auth"],
        "time_start": "2023-01-01T00:00:00",
    })
    assert "excluded" not in [r.chunk_id for r in results]
