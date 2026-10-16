"""Offline reproductions for PROJECT_AUDIT.md; asserts observed defects.

Run: python docs/audit_checks.py
These are diagnostic checks, not regression tests of desired behavior.
Leaf modules are loaded without package __init__ imports so unrelated missing
Qdrant/RAGAS dependencies do not prevent independent local checks.
SQLite and embedding-cache checks use memory, never the project corpus.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(name: str) -> None:
    module = ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split('.')))]
    sys.modules[name] = module


def leaf(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT.joinpath(*name.split('.')).with_suffix('.py')
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    for name in ('indexing', 'retrieval', 'evaluation', 'evaluation.custom_metrics'):
        package(name)
    from config.settings import settings
    settings.gemini_api_key = None  # no remote calls or credentials
    from parsing.multi_schema_parser import MultiSchemaParser
    from chunking.commit_chunker import CommitChunker
    from chunking.changelog_chunker import ChangelogChunker
    from embedding.models import OllamaEmbeddingClient
    from embedding.embedding_cache import EmbeddingCache
    from embedding.embedder import DualEmbedder
    from generation.llm_client import LLMClient
    fts_module = leaf('indexing.fts_index')
    types = leaf('retrieval.types')
    context_module = leaf('retrieval.context_assembler')
    temporal = leaf('evaluation.custom_metrics.temporal_accuracy')
    entity = leaf('evaluation.custom_metrics.entity_consistency')
    parser = MultiSchemaParser()
    commit = parser.parse_commit({
        'sha': 'abc', 'message': 'Subject\nCritical second line\nThird line',
        'authored_at': '2023-01-01T00:00:00+00:00',
        'file_paths': ['src/auth.py'],
    })
    first = CommitChunker().chunk(commit, repo='owner/repo')[0]
    second = CommitChunker().chunk(commit, repo='owner/repo')[0]
    assert first.metadata.chunk_id != second.metadata.chunk_id
    print('CONFIRMED: identical input generates different chunk IDs')
    assert 'Critical second line' not in first.text
    print('CONFIRMED: commit body without blank separator loses second line')
    assert first.metadata.module_tags == [] and first.metadata.entity_tags == []
    print('CONFIRMED: commit chunk receives no module/entity tags')
    release = parser.parse_release({
        'tag': 'v1.0', 'published_at': '2023-01-01T00:00:00+00:00',
        'body': '## v1.0\n' + 'x' * 6000,
    })
    release_chunks = ChangelogChunker().chunk(release)
    assert len(release_chunks) > 1
    assert len({c.metadata.chunk_id for c in release_chunks}) == 1
    print('CONFIRMED: split release chunks all share one chunk ID')

    fts = fts_module.FTSIndex(':memory:')
    for c in (first, second):
        fts.add_chunks([{'text': c.text, 'metadata': c.metadata.to_dict()}])
    assert fts.count() == 2
    fts.close()
    print('CONFIRMED: indexing the same logical chunk twice accumulates FTS records')

    client = OllamaEmbeddingClient()
    client._client.close()
    client._client = Mock()
    client._client.post.side_effect = RuntimeError('simulated outage')
    client._gemini = None
    cache = EmbeddingCache.__new__(EmbeddingCache)
    cache._cache = {}
    cache.close = Mock()  # avoid persistence; exercise real get/put/batch methods
    embedder = DualEmbedder(client=client, cache=cache)
    vector = embedder.embed_chunks([
        {'text': 'content', 'metadata': {'chunk_id': 'a', 'doc_type': 'commit'}}
    ], show_progress=False)[0].semantic_vector
    assert len(vector) == 768 and not any(vector)
    assert cache.get('content', embedder.prose_model) == vector
    embedder.close()
    print('CONFIRMED: total embedding outage returns and caches zero vector')

    llm = LLMClient()
    llm._http.close()
    llm._http = Mock()
    llm._http.post.side_effect = RuntimeError('simulated outage')
    result = llm.generate('test')
    assert result.text == ''
    llm.close()
    print('CONFIRMED: total generation outage returns empty successful result')

    assembler = context_module.ContextAssembler.__new__(context_module.ContextAssembler)
    assembler.max_tokens = 20
    assembler.chars_per_token = 4
    assembler._tokenizer = None  # deterministic local fallback, no tokenizer download
    candidates = [
        types.CandidateChunk('large', 1.0, 'dense', 'x' * 500, {}),
        types.CandidateChunk('small', 0.5, 'dense', 'short evidence', {}),
    ]
    assert assembler.assemble(candidates) == ''
    print('CONFIRMED: oversized first candidate prevents all subsequent context')
    assert temporal.TemporalAccuracyMetric().score(
        ['It happened in 2023 and 1999'], [['2023-01-01']]
    ) == 1.0
    assert entity.EntityConsistencyMetric().score(
        ['JWT uses Redis and Kafka'], [[['JWT']]]
    ) == 1.0
    print('CONFIRMED: unsupported years/entities still receive perfect custom scores')
    gql = parser.parse_pr_graphql({'number': 42, 'body': 'Change', 'closing_issues': ['#7: Motivation']})
    from chunking.log_chunker import LogChunker
    assert '#7' not in LogChunker('pr_graphql').chunk(gql)[0].text
    print('CONFIRMED: GraphQL closing issue references omitted from chunk text')

    # Exercise actual fusion with fake retrieval backends. The dense backend
    # correctly returns no matching documents; sparse supplies an excluded one.
    qdrant_stub = ModuleType('indexing.qdrant_store')
    qdrant_stub.QdrantStore = SimpleNamespace(build_filter=lambda **kwargs: kwargs)
    sys.modules['indexing.qdrant_store'] = qdrant_stub
    leaf('indexing.bm25_index')
    leaf('retrieval.chunk_store')
    leaf('retrieval.query_embedder')
    hybrid = leaf('retrieval.hybrid_retriever')
    excluded = {'text': 'Excluded issue', 'metadata': {
        'chunk_id': 'excluded', 'doc_type': 'issue', 'author': 'bob',
        'timestamp': '1999-01-01', 'module_tags': ['payments'],
    }}
    retriever = hybrid.HybridRetriever(
        qdrant=SimpleNamespace(search_semantic=lambda **kwargs: []),
        bm25=SimpleNamespace(search=lambda *args, **kwargs: [SimpleNamespace(chunk_id='excluded')]),
        fts=SimpleNamespace(search=lambda *args, **kwargs: []),
        chunk_store=SimpleNamespace(get=lambda chunk_id: excluded),
        embedder=SimpleNamespace(embed=lambda text: [1.0]),
    )
    results = retriever.retrieve('test', filters={
        'doc_types': ['commit'], 'author': 'alice', 'module_tags': ['auth'],
        'time_start': '2023-01-01',
    })
    assert results[0].chunk_id == 'excluded'
    print('CONFIRMED: sparse fusion returns a candidate excluded by every requested filter')
    print('Completed 11 offline defect checks; no live providers used.')


if __name__ == '__main__':
    main()
