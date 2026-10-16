"""qdrant_store.py — Qdrant collection management (semantic vector + optional code vector).

Per-generation collections
--------------------------
Each index generation owns its own collection (``collection=<gen.qdrant_collection>``).
The collection's metadata records the embedding identity, the generation id,
the repo and whether the optional ``code`` named vector exists, so a query
embedder can verify it is in the same vector space (``verify_embedding``).

Local (on-disk) mode
--------------------
Qdrant local mode allows **one process per storage path**.  A second opener
gets a ``QdrantLockError`` (it does not hang).  Use a Qdrant server
(``QDRANT_URL``) when several processes (e.g. the API and an indexer) must
share data.  Several collections *can* live under one local path, which is how
generations are laid out.

Vectors
-------
Zero, non-finite or wrong-dimension vectors are rejected at upsert time with
``InvalidEmbeddingError`` (never silently skipped or zero-filled).  The ``code``
named vector exists only when ``dual_vectors`` is on, i.e. when a code model
that differs from the prose model is actually configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    DatetimeRange,
    PayloadSchemaType,
)

from embedding.models import (
    PROSE_DIM,
    CODE_DIM,
    EmbeddingIdentity,
    EmbeddingIdentityMismatchError,
    InvalidEmbeddingError,
    validate_vector,
)


COLLECTION_NAME = "gitmind_chunks"


class QdrantLockError(RuntimeError):
    """Another process/client already owns this local Qdrant storage path."""


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


def _to_utc(value: Any) -> datetime | None:
    """Datetime or ISO string -> aware UTC datetime (naive is treated as UTC)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class QdrantStore:
    """Manages a Qdrant collection with named vectors 'semantic' (+ 'code').

    Connects to a remote Qdrant (Cloud or a local docker service) whenever
    QDRANT_URL is configured; QDRANT_API_KEY is optional. Local on-disk mode is
    used only when no URL is set (single owner per path, see module docstring).

    ``dual_vectors``: True creates the ``code`` named vector, False omits it,
    None (default) means "False when creating, whatever exists when opening".
    """

    def __init__(
        self,
        path: str = ".qdrant",
        collection_name: str = COLLECTION_NAME,
        semantic_dim: int = PROSE_DIM,
        code_dim: int = CODE_DIM,
        *,
        collection: str | None = None,
        dual_vectors: bool | None = None,
        embedding: dict | None = None,
        generation_id: str = "",
        repo: str = "",
    ) -> None:
        self.collection_name = collection or collection_name
        self.semantic_dim = semantic_dim
        self.code_dim = code_dim
        self.dual_vectors = dual_vectors
        self.embedding = embedding
        self.generation_id = generation_id
        self.repo = repo
        self._code_enabled_cache: bool | None = dual_vectors

        from config.settings import settings as _s
        # The remote client drops collection metadata (qdrant-client accepts it but never
        # sends it), so remote stores keep the identity in a companion collection.
        self._remote = bool(_s.qdrant_url)
        if _s.qdrant_url:
            # Remote service; the API key is optional (e.g. unauthenticated docker).
            self.client = QdrantClient(
                url=_s.qdrant_url,
                api_key=_s.qdrant_api_key or None,
            )
        else:
            # Development: local on-disk storage (one owner per path)
            try:
                self.client = QdrantClient(path=path)
            except RuntimeError as exc:
                if "already accessed" in str(exc):
                    raise QdrantLockError(
                        f"local Qdrant storage {path!r} is already open in another "
                        "process/client; local mode allows a single owner. Stop the "
                        "other process or use a Qdrant server (QDRANT_URL)."
                    ) from exc
                raise

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist yet."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if self.embedding is not None:
                self.verify_embedding(self.embedding)
            return

        vectors_config = {
            "semantic": VectorParams(size=self.semantic_dim, distance=Distance.COSINE)
        }
        if self.dual_vectors:
            vectors_config["code"] = VectorParams(size=self.code_dim, distance=Distance.COSINE)
        self._code_enabled_cache = bool(self.dual_vectors)

        meta = {
            "embedding": self.embedding or {},
            "generation_id": self.generation_id,
            "repo": self.repo,
            "dual_vectors": bool(self.dual_vectors),
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            metadata=meta,
        )
        if self._remote:
            self._write_companion_metadata(meta)

        # Create payload indexes for fast filtering (no-ops in local mode)
        for field in ("doc_type", "author", "graph_node_id", "repo", "scope",
                      "commit_sha", "module_tags"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="timestamp",
            field_schema=PayloadSchemaType.DATETIME,
        )

    # ------------------------------------------------------------------
    # Collection metadata / identity
    # ------------------------------------------------------------------

    @property
    def _companion_name(self) -> str:
        return f"{self.collection_name}__meta"

    def _write_companion_metadata(self, meta: dict) -> None:
        import uuid as _uuid
        from qdrant_client.models import Distance, PointStruct, VectorParams

        name = self._companion_name
        if name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=name, vectors_config=VectorParams(size=1, distance=Distance.DOT)
            )
        self.client.upsert(
            collection_name=name,
            points=[PointStruct(id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, "metadata")),
                                vector=[0.0], payload={"metadata": meta})],
            wait=True,
        )

    def get_metadata(self) -> dict:
        """Metadata stored with the collection (embedding identity, generation...)."""
        if self._remote:
            name = self._companion_name
            if name not in [c.name for c in self.client.get_collections().collections]:
                return {}
            points, _ = self.client.scroll(collection_name=name, limit=1, with_payload=True)
            return dict((points[0].payload or {}).get("metadata") or {}) if points else {}
        info = self.client.get_collection(self.collection_name)
        return dict(getattr(info.config, "metadata", None) or {})

    def verify_embedding(self, expected: dict) -> None:
        """Raise EmbeddingIdentityMismatchError unless the collection's recorded
        semantic identity matches *expected* (a dict with a 'semantic' identity
        dict, as stored in the generation manifest)."""
        stored = (self.get_metadata().get("embedding") or {}).get("semantic")
        wanted = (expected or {}).get("semantic")
        if not stored or not wanted:
            raise EmbeddingIdentityMismatchError(
                f"collection {self.collection_name!r} has no recorded embedding identity"
            )
        if not EmbeddingIdentity.from_dict(stored).matches(EmbeddingIdentity.from_dict(wanted)):
            raise EmbeddingIdentityMismatchError(
                f"collection {self.collection_name!r} was built with "
                f"{EmbeddingIdentity.from_dict(stored).key}, expected "
                f"{EmbeddingIdentity.from_dict(wanted).key}"
            )

    def _code_enabled(self) -> bool:
        if self._code_enabled_cache is None:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            self._code_enabled_cache = isinstance(vectors, dict) and "code" in vectors
        return self._code_enabled_cache

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    @staticmethod
    def _payload(ec, default_repo: str, generation_id: str) -> dict:
        meta = dict(ec.metadata) if ec.metadata else {}
        ts = _to_utc(meta.get("timestamp"))
        payload = {
            **meta,
            "chunk_id": ec.chunk_id,
            "text": ec.text,
            "repo": meta.get("repo") or default_repo,
            "doc_type": meta.get("doc_type") or getattr(ec, "doc_type", "") or "",
            "doc_id": meta.get("doc_id", ""),
            "author": meta.get("author", ""),
            "timestamp": ts.isoformat() if ts else "",
            "module_tags": list(meta.get("module_tags") or []),
            "entity_tags": list(meta.get("entity_tags") or []),
            "scope": meta.get("scope") or "history",
            "commit_sha": meta.get("commit_sha", ""),
            "file_paths": list(meta.get("file_paths") or []),
        }
        if generation_id:
            payload["generation_id"] = generation_id
        return payload

    def upsert_embedded_chunks(self, embedded_chunks: list) -> int:
        """Upsert EmbeddedChunk objects into Qdrant.

        Each point has the named vector 'semantic' and, when dual vectors are
        enabled and the chunk has one, 'code'.  Invalid vectors (zero, NaN,
        wrong dimension, missing semantic vector) raise InvalidEmbeddingError
        before anything is written.
        """
        code_enabled = self._code_enabled()
        points: list[PointStruct] = []
        for ec in embedded_chunks:
            try:
                if not ec.semantic_vector:
                    raise InvalidEmbeddingError("missing semantic vector")
                vectors = {"semantic": validate_vector(ec.semantic_vector, self.semantic_dim)}
                if code_enabled and ec.code_vector:
                    vectors["code"] = validate_vector(ec.code_vector, self.code_dim)
            except InvalidEmbeddingError as exc:
                raise InvalidEmbeddingError(f"chunk {ec.chunk_id}: {exc}") from exc

            points.append(
                PointStruct(
                    id=_chunk_id_to_uuid(ec.chunk_id),
                    vector=vectors,
                    payload=self._payload(ec, self.repo, self.generation_id),
                )
            )

        # Upsert in batches of 100
        for start in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + 100],
            )
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _as_filter(query_filter) -> Filter | None:
        """Accept a qdrant Filter, a Scope, or None."""
        if query_filter is None or isinstance(query_filter, Filter):
            return query_filter
        if hasattr(query_filter, "repo") and hasattr(query_filter, "matches"):
            return QdrantStore.build_filter(query_filter)
        raise TypeError(f"unsupported filter type {type(query_filter).__name__}")

    def _search(self, using: str, query_vector, limit, query_filter) -> list[SearchResult]:
        # The filter is evaluated by Qdrant *before* the limit is applied.
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=using,
            limit=limit,
            query_filter=self._as_filter(query_filter),
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

    def search_semantic(
        self,
        query_vector: list[float],
        limit: int = 20,
        query_filter: Filter | None = None,
    ) -> list[SearchResult]:
        """Dense search using the 'semantic' named vector."""
        return self._search("semantic", query_vector, limit, query_filter)

    def search_code(
        self,
        query_vector: list[float],
        limit: int = 20,
        query_filter: Filter | None = None,
    ) -> list[SearchResult]:
        """Dense search using the 'code' named vector ([] if there is none)."""
        if not self._code_enabled():
            return []
        return self._search("code", query_vector, limit, query_filter)

    # ------------------------------------------------------------------
    # Filters (convenience builders)
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        scope=None,
        *,
        doc_types: list[str] | None = None,
        module_tags: list[str] | None = None,
        author: str | None = None,
        time_start: Any = None,
        time_end: Any = None,
        repo: str | None = None,
        kind: str | None = None,
        commit: str | None = None,
        authors: list[str] | None = None,
    ) -> Filter | None:
        """Build a Qdrant Filter from a ``retrieval.scope.Scope`` and/or criteria.

        Mirrors ``Scope.matches``: repo, doc types, authors, modules (any
        overlap with module_tags), kind (payload ``scope``), commit
        (``commit_sha``) and an inclusive time range on ``timestamp``.
        """
        if scope is not None:
            repo = scope.repo
            doc_types = list(scope.doc_types) or doc_types
            module_tags = list(scope.modules) or module_tags
            authors = list(scope.authors) or authors
            kind = scope.kind
            commit = scope.commit or commit
            time_start = scope.time_start or time_start
            time_end = scope.time_end or time_end

        conditions = []
        if repo:
            conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))
        if doc_types:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchAny(any=list(doc_types)))
            )
        if module_tags:
            conditions.append(
                FieldCondition(key="module_tags", match=MatchAny(any=list(module_tags)))
            )
        if author:
            conditions.append(
                FieldCondition(key="author", match=MatchValue(value=author))
            )
        if authors:
            conditions.append(
                FieldCondition(key="author", match=MatchAny(any=list(authors)))
            )
        if kind:
            conditions.append(FieldCondition(key="scope", match=MatchValue(value=kind)))
        if commit:
            conditions.append(FieldCondition(key="commit_sha", match=MatchValue(value=commit)))
        if time_start or time_end:
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=DatetimeRange(
                        gte=_to_utc(time_start),
                        lte=_to_utc(time_end),
                    ),
                )
            )
        return Filter(must=conditions) if conditions else None

    # ------------------------------------------------------------------
    # Collection management / info
    # ------------------------------------------------------------------

    def delete_collection(self, name: str | None = None) -> bool:
        """Delete a collection (default: this store's).  Returns True if it existed."""
        name = name or self.collection_name
        existed = name in [c.name for c in self.client.get_collections().collections]
        if existed:
            self.client.delete_collection(collection_name=name)
        if self._remote and name == self.collection_name:
            companion = f"{name}__meta"
            if companion in [c.name for c in self.client.get_collections().collections]:
                self.client.delete_collection(collection_name=companion)
        return existed

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def close(self) -> None:
        self.client.close()
