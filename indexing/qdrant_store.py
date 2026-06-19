"""qdrant_store.py — Qdrant collection management with dual named vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    DatetimeRange,
    PayloadSchemaType,
)

from embedding.models import PROSE_DIM, CODE_DIM


COLLECTION_NAME = "gitmind_chunks"


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    payload: dict


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert an arbitrary string chunk_id to a deterministic UUID5.

    Qdrant requires point IDs to be UUIDs or unsigned ints, not arbitrary
    strings.  We derive a stable UUID from the chunk_id so that re-ingestion
    is idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantStore:
    """Manages a Qdrant collection with named vectors: 'semantic' + 'code'.

    Connects to Qdrant Cloud when QDRANT_URL + QDRANT_API_KEY are configured,
    otherwise falls back to a local on-disk mode for development.
    """

    def __init__(
        self,
        path: str = ".qdrant",
        collection_name: str = COLLECTION_NAME,
        semantic_dim: int = PROSE_DIM,
        code_dim: int = CODE_DIM,
    ) -> None:
        self.collection_name = collection_name
        self.semantic_dim = semantic_dim
        self.code_dim = code_dim

        from config.settings import settings as _s
        if _s.qdrant_url and _s.qdrant_api_key:
            # Production: connect to Qdrant Cloud
            self.client = QdrantClient(
                url=_s.qdrant_url,
                api_key=_s.qdrant_api_key,
            )
        else:
            # Development: local on-disk storage
            self.client = QdrantClient(path=path)

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist yet."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "semantic": VectorParams(
                    size=self.semantic_dim, distance=Distance.COSINE
                ),
                "code": VectorParams(
                    size=self.code_dim, distance=Distance.COSINE
                ),
            },
        )

        # Create payload indexes for fast filtering
        for field in ("doc_type", "author", "graph_node_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_embedded_chunks(self, embedded_chunks: list) -> int:
        """Upsert EmbeddedChunk objects into Qdrant.

        Each point has named vectors 'semantic' and optionally 'code'.
        Payload contains full chunk metadata + text.
        """
        points: list[PointStruct] = []
        for ec in embedded_chunks:
            vectors = {}
            if ec.semantic_vector:
                vectors["semantic"] = ec.semantic_vector
            if ec.code_vector:
                vectors["code"] = ec.code_vector

            if not vectors:
                continue

            # If only semantic exists, fill code with zeros so the point schema
            # stays consistent (Qdrant requires all named vectors per point)
            if "code" not in vectors:
                vectors["code"] = [0.0] * self.code_dim
            if "semantic" not in vectors:
                vectors["semantic"] = [0.0] * self.semantic_dim

            payload = dict(ec.metadata) if ec.metadata else {}
            payload["text"] = ec.text
            # Always store the original string chunk_id in the payload
            payload["chunk_id"] = ec.chunk_id

            # Qdrant only accepts UUID or unsigned-int point IDs.
            point_uuid = _chunk_id_to_uuid(ec.chunk_id)

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=vectors,
                    payload=payload,
                )
            )

        if points:
            # Upsert in batches of 100
            for start in range(0, len(points), 100):
                batch = points[start : start + 100]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query_vector: list[float],
        limit: int = 20,
        query_filter: Filter | None = None,
    ) -> list[SearchResult]:
        """Dense search using the 'semantic' named vector."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="semantic",
            limit=limit,
            query_filter=query_filter,
        ).points
        return [
            SearchResult(
                # Prefer original string chunk_id stored in payload; fall back to UUID str.
                chunk_id=str(r.payload.get("chunk_id", r.id)) if r.payload else str(r.id),
                score=r.score,
                payload=r.payload or {},
            )
            for r in results
        ]

    def search_code(
        self,
        query_vector: list[float],
        limit: int = 20,
        query_filter: Filter | None = None,
    ) -> list[SearchResult]:
        """Dense search using the 'code' named vector."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="code",
            limit=limit,
            query_filter=query_filter,
        ).points
        return [
            SearchResult(
                chunk_id=str(r.payload.get("chunk_id", r.id)) if r.payload else str(r.id),
                score=r.score,
                payload=r.payload or {},
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Filters (convenience builders)
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        doc_types: list[str] | None = None,
        module_tags: list[str] | None = None,
        author: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> Filter | None:
        """Build a Qdrant Filter from common criteria."""
        conditions = []
        if doc_types:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchAny(any=doc_types))
            )
        if module_tags:
            conditions.append(
                FieldCondition(key="module_tags", match=MatchAny(any=module_tags))
            )
        if author:
            conditions.append(
                FieldCondition(key="author", match=MatchValue(value=author))
            )
        if time_start or time_end:
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=DatetimeRange(
                        gte=time_start,
                        lte=time_end,
                    ),
                )
            )
        return Filter(must=conditions) if conditions else None

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def close(self) -> None:
        self.client.close()
