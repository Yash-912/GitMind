from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Iterable

from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex

from .scope import to_utc
from .types import CandidateChunk
from .chunk_store import ChunkStore
from .query_embedder import QueryEmbedder

PERSON_BOOST = 1.15


def _legacy_matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Same predicate as Scope.matches, for the unscoped ``filters=`` dict."""
    doc_types = filters.get("doc_types")
    if doc_types and meta.get("doc_type") not in doc_types:
        return False
    author = filters.get("author")
    if author and meta.get("author") != author:
        return False
    module_tags = filters.get("module_tags")
    if module_tags and not set(module_tags) & set(meta.get("module_tags") or []):
        return False
    if filters.get("time_start") or filters.get("time_end"):
        ts = to_utc(meta.get("timestamp"))
        if ts is None:
            return False
        start, end = to_utc(filters.get("time_start")), to_utc(filters.get("time_end"))
        if start and ts < start:
            return False
        if end and ts > end:
            return False
    return True


class HybridRetriever:
    """Combine dense, sparse, and FTS retrieval with RRF fusion.

    With a ``scope`` the same scope constrains every branch: allowed chunk ids
    (computed once per scope and cached) are passed to BM25/FTS, a Qdrant
    filter is built from the scope, and after fusion every candidate is
    re-checked with ``scope.matches`` so a branch that ignores its filter can
    never leak an out-of-scope chunk. Without a scope the legacy ``filters``
    dict behaviour is unchanged.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        bm25: BM25Index,
        fts: FTSIndex,
        chunk_store: ChunkStore,
        embedder: QueryEmbedder | None = None,
        rrf_k: int = 60,
        use_code_search: bool = True,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.bm25 = bm25
        self.fts = fts
        self.chunk_store = chunk_store
        self.embedder = embedder or QueryEmbedder()
        self.rrf_k = rrf_k
        self.use_code_search = use_code_search
        self.source_weights = source_weights or {
            "qdrant_semantic": 1.0,
            "qdrant_code": 0.8,
            "bm25": 0.7,
            "fts": 0.6,
        }
        self._allowed_cache: dict[Any, frozenset[str]] = {}

    # ------------------------------------------------------------------ scope

    def allowed_ids_for(self, scope: Any) -> frozenset[str]:
        """Chunk ids satisfying `scope`; computed once per scope."""
        try:
            return self._allowed_cache[scope]
        except KeyError:
            ids = frozenset(scope.filter_ids(self.chunk_store.records()))
            self._allowed_cache[scope] = ids
            return ids

    def _scope_filter(self, scope: Any) -> Any:
        build = getattr(self.qdrant, "build_filter", None)
        if build is None:
            return None
        try:
            params = list(inspect.signature(build).parameters)
        except (TypeError, ValueError):
            params = []
        if params and params[0] == "scope":
            return build(scope)
        # Legacy keyword builder: cannot express repo/kind/commit or several
        # authors, so the post-fusion scope check enforces those.
        return build(
            doc_types=list(scope.doc_types) or None,
            module_tags=list(scope.modules) or None,
            author=scope.authors[0] if len(scope.authors) == 1 else None,
            time_start=scope.time_start,
            time_end=scope.time_end,
        )

    # --------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        limit: int = 40,
        filters: dict[str, Any] | None = None,
        scope: Any = None,
        allowed_ids: Iterable[str] | None = None,
        doc_type_weights: dict[str, float] | None = None,
        boost_terms: Iterable[str] = (),
    ) -> list[CandidateChunk]:
        return self.retrieve_many(
            [query], limit, filters, scope, allowed_ids, doc_type_weights, boost_terms
        )

    def retrieve_many(
        self,
        queries: Iterable[str],
        limit: int = 40,
        filters: dict[str, Any] | None = None,
        scope: Any = None,
        allowed_ids: Iterable[str] | None = None,
        doc_type_weights: dict[str, float] | None = None,
        boost_terms: Iterable[str] = (),
    ) -> list[CandidateChunk]:
        """Retrieve every query separately (same scope) and RRF-fuse all branch lists."""
        allowed: frozenset[str] | None = None
        if scope is not None:
            allowed = frozenset(allowed_ids) if allowed_ids is not None else self.allowed_ids_for(scope)
            if not allowed:
                return []
            q_filter = self._scope_filter(scope)
        else:
            q_filter = QdrantStore.build_filter(
                doc_types=filters.get("doc_types") if filters else None,
                module_tags=filters.get("module_tags") if filters else None,
                author=filters.get("author") if filters else None,
                time_start=filters.get("time_start") if filters else None,
                time_end=filters.get("time_end") if filters else None,
            )
        def sparse(index: Any, query: str) -> list[Any]:
            """Sparse branch; allowed ids are applied before the limit (filter-then-rank)."""
            if allowed is None:
                return index.search(query, limit=limit)
            return index.search(query, limit=limit, allowed_ids=allowed)

        branch_lists: list[tuple[str, list[Any]]] = []
        for query in dict.fromkeys(q for q in queries if q and q.strip()):
            vec = self.embedder.embed(query)  # typed EmbeddingError propagates
            branch_lists.append(
                ("qdrant_semantic",
                 self.qdrant.search_semantic(query_vector=vec, limit=limit, query_filter=q_filter))
            )
            if self.use_code_search and hasattr(self.qdrant, "search_code"):
                branch_lists.append(
                    ("qdrant_code",
                     self.qdrant.search_code(query_vector=vec, limit=limit, query_filter=q_filter))
                )
            branch_lists.append(("bm25", sparse(self.bm25, query)))
            branch_lists.append(("fts", sparse(self.fts, query)))

        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, str] = {}
        for source, results in branch_lists:
            w = self.source_weights.get(source, 1.0)
            for rank, r in enumerate(results):
                scores[r.chunk_id] += w / (self.rrf_k + rank + 1)
                sources.setdefault(r.chunk_id, source)

        boosts = [t.lower() for t in boost_terms if t]
        weights = doc_type_weights or {}
        candidates: list[CandidateChunk] = []
        for chunk_id, score in scores.items():
            record = self.chunk_store.get(chunk_id)
            if not record:
                continue
            meta = record.get("metadata", {})
            # Authoritative final check: a branch that ignored its filter cannot leak.
            if scope is not None and not scope.matches(meta):
                continue
            if scope is None and filters and not _legacy_matches(meta, filters):
                continue
            score *= weights.get(meta.get("doc_type", ""), 1.0)
            if boosts:
                hay = f"{meta.get('author', '')} {record.get('text', '')}".lower()
                if any(t in hay for t in boosts):
                    score *= PERSON_BOOST
            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    score=score,
                    source=sources.get(chunk_id, "rrf"),
                    text=record.get("text", ""),
                    metadata=meta,
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.chunk_id))
        return candidates[:limit]

    def close(self) -> None:
        self.embedder.close()
