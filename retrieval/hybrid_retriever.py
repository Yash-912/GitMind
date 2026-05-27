from __future__ import annotations

from collections import defaultdict
from typing import Any

from indexing.qdrant_store import QdrantStore
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex

from .types import CandidateChunk
from .chunk_store import ChunkStore
from .query_embedder import QueryEmbedder


class HybridRetriever:
    """Combine dense, sparse, and FTS retrieval with RRF fusion."""

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

    def retrieve(
        self,
        query: str,
        limit: int = 40,
        filters: dict[str, Any] | None = None,
    ) -> list[CandidateChunk]:
        query_vector = self.embedder.embed(query)
        q_filter = QdrantStore.build_filter(
            doc_types=filters.get("doc_types") if filters else None,
            module_tags=filters.get("module_tags") if filters else None,
            author=filters.get("author") if filters else None,
            time_start=filters.get("time_start") if filters else None,
            time_end=filters.get("time_end") if filters else None,
        )

        dense_results = self.qdrant.search_semantic(
            query_vector=query_vector,
            limit=limit,
            query_filter=q_filter,
        )
        code_results = []
        if self.use_code_search and hasattr(self.qdrant, "search_code"):
            code_results = self.qdrant.search_code(
                query_vector=query_vector,
                limit=limit,
                query_filter=q_filter,
            )
        bm25_results = self.bm25.search(query, limit=limit)
        fts_results = self.fts.search(query, limit=limit)

        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, str] = {}

        for rank, r in enumerate(dense_results):
            w = self.source_weights.get("qdrant_semantic", 1.0)
            scores[r.chunk_id] += w / (self.rrf_k + rank + 1)
            sources.setdefault(r.chunk_id, "qdrant_semantic")

        for rank, r in enumerate(code_results):
            w = self.source_weights.get("qdrant_code", 1.0)
            scores[r.chunk_id] += w / (self.rrf_k + rank + 1)
            sources.setdefault(r.chunk_id, "qdrant_code")

        for rank, r in enumerate(bm25_results):
            w = self.source_weights.get("bm25", 0.7)
            scores[r.chunk_id] += w / (self.rrf_k + rank + 1)
            sources.setdefault(r.chunk_id, "bm25")

        for rank, r in enumerate(fts_results):
            w = self.source_weights.get("fts", 0.6)
            scores[r.chunk_id] += w / (self.rrf_k + rank + 1)
            sources.setdefault(r.chunk_id, "fts")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        candidates: list[CandidateChunk] = []
        for chunk_id, score in ranked:
            record = self.chunk_store.get(chunk_id)
            if not record:
                continue
            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    score=score,
                    source=sources.get(chunk_id, "rrf"),
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                )
            )
        return candidates

    def close(self) -> None:
        self.embedder.close()
