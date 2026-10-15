"""embedder.py — Dual embedding orchestrator with caching, batching and identity pinning.

* One ``EmbeddingIdentity`` is pinned per vector kind (semantic / code) for the
  whole embedder, so cached vectors and fresh vectors can never come from
  different providers.
* Failures are reported per chunk; ``embed_chunks`` raises instead of
  returning a partial result unless ``allow_partial=True``.
* A separate code vector is only produced when the code model actually differs
  from the prose model (``code_model_is_distinct``).  With the default
  settings both are ``nomic-embed-text``, so diff/code chunks get a semantic
  vector only and no code-embedding benefit is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    OllamaEmbeddingClient,
    EmbeddingBatch,
    EmbeddingBatchError,
    EmbeddingFailure,
    EmbeddingIdentity,
    InvalidEmbeddingError,
    PROSE_MODEL,
    CODE_MODEL,
    code_model_is_distinct,
)
from .embedding_cache import EmbeddingCache


@dataclass
class EmbeddedChunk:
    """A chunk with its semantic and/or code embedding vectors attached."""
    chunk_id: str
    text: str
    semantic_vector: list[float] | None = None
    code_vector: list[float] | None = None
    doc_type: str = ""
    metadata: dict | None = None
    semantic_identity: EmbeddingIdentity | None = None
    code_identity: EmbeddingIdentity | None = None


@dataclass(frozen=True)
class ChunkEmbeddingFailure:
    """A chunk (or one of its vectors) that could not be embedded."""
    chunk_id: str
    vector_kind: str  # "semantic" | "code"
    error_type: str
    message: str


@dataclass
class EmbeddingReport:
    """Outcome of an embedding run.  ``chunks`` holds only fully embedded chunks."""
    chunks: list[EmbeddedChunk]
    failures: list[ChunkEmbeddingFailure] = field(default_factory=list)
    semantic_identity: EmbeddingIdentity | None = None
    code_identity: EmbeddingIdentity | None = None
    dual_vectors: bool = False
    cache_hits: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures


# Doc types that should get code-specific embeddings
_CODE_DOC_TYPES = {"diff", "code"}


class DualEmbedder:
    """Embed chunks with the prose model and, if distinct, a code model.

    Every chunk gets a semantic embedding.  Chunks with doc_type in
    {"diff", "code"} also get a code embedding *only if* the code model differs
    from the prose model.

    Uses EmbeddingCache (keyed by actual provider/model/dim) to skip
    already-embedded chunks on re-ingestion.
    """

    def __init__(
        self,
        client: OllamaEmbeddingClient | None = None,
        cache: EmbeddingCache | None = None,
        prose_model: str = PROSE_MODEL,
        code_model: str = CODE_MODEL,
        batch_size: int = 32,
        expected_semantic_identity: EmbeddingIdentity | None = None,
        expected_code_identity: EmbeddingIdentity | None = None,
    ) -> None:
        self.client = client or OllamaEmbeddingClient()
        self.cache = cache or EmbeddingCache()
        self.prose_model = prose_model
        self.code_model = code_model
        self.batch_size = batch_size
        self.expected_semantic_identity = expected_semantic_identity
        self.expected_code_identity = expected_code_identity
        self.semantic_identity: EmbeddingIdentity | None = None
        self.code_identity: EmbeddingIdentity | None = None

    @property
    def dual_vectors_enabled(self) -> bool:
        """True only when a genuinely different code model is configured."""
        return code_model_is_distinct(self.prose_model, self.code_model)

    # ------------------------------------------------------------------
    # Identity pinning
    # ------------------------------------------------------------------

    def resolve_identities(self) -> tuple[EmbeddingIdentity, EmbeddingIdentity | None]:
        """Pin (once) the identities used for all later embedding calls.

        Raises EmbeddingUnavailableError / EmbeddingIdentityMismatchError.
        """
        if self.semantic_identity is None:
            self.semantic_identity = self.client.resolve_identity(
                self.prose_model, expected=self.expected_semantic_identity
            )
        if self.dual_vectors_enabled and self.code_identity is None:
            self.code_identity = self.client.resolve_identity(
                self.code_model, expected=self.expected_code_identity
            )
        return self.semantic_identity, self.code_identity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str], model: str | None = None) -> EmbeddingBatch:
        """Cache-aware ``embed_texts`` for the prose (default) or code model."""
        model = model or self.prose_model
        self.resolve_identities()
        identity = (
            self.code_identity
            if self.dual_vectors_enabled and model == self.code_model
            else self.semantic_identity
        )
        return self._embed_with_cache(texts, model, identity)  # type: ignore[arg-type]

    def embed_chunks_report(
        self,
        chunks: list[dict],
        show_progress: bool = True,
    ) -> EmbeddingReport:
        """Embed chunk dicts and report successes and per-chunk failures."""
        if not chunks:
            return EmbeddingReport(chunks=[], dual_vectors=self.dual_vectors_enabled)

        semantic_id, code_id = self.resolve_identities()
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        chunk_ids = [m["chunk_id"] for m in metas]
        doc_types = [m.get("doc_type", "") for m in metas]
        failures: list[ChunkEmbeddingFailure] = []
        bad: set[int] = set()
        cache_hits = 0

        # ---- Semantic embeddings (all chunks) ----
        if show_progress:
            print(f"  Embedding {len(texts)} chunks with {semantic_id.key}...")
        sem = self._embed_with_cache(texts, self.prose_model, semantic_id)
        cache_hits += sem.cache_hits
        for f in sem.failures:
            failures.append(ChunkEmbeddingFailure(chunk_ids[f.index], "semantic", f.error_type, f.message))
            bad.add(f.index)

        # ---- Code embeddings (diff/code chunks, only with a distinct model) ----
        code_vectors: dict[int, list[float]] = {}
        code_indices = [i for i, dt in enumerate(doc_types) if dt in _CODE_DOC_TYPES]
        if code_indices and code_id is not None:
            code_texts = [texts[i] for i in code_indices]
            if show_progress:
                print(f"  Embedding {len(code_texts)} code chunks with {code_id.key}...")
            code = self._embed_with_cache(code_texts, self.code_model, code_id)
            cache_hits += code.cache_hits
            for f in code.failures:
                idx = code_indices[f.index]
                failures.append(ChunkEmbeddingFailure(chunk_ids[idx], "code", f.error_type, f.message))
                bad.add(idx)
            for idx, vec in zip(code_indices, code.vectors):
                if vec is not None:
                    code_vectors[idx] = vec
        elif code_indices and show_progress:
            print(
                f"  Code model == prose model ({self.prose_model}); skipping separate "
                f"code vectors for {len(code_indices)} diff/code chunks."
            )

        # ---- Assemble results (only fully embedded chunks) ----
        results: list[EmbeddedChunk] = []
        for i in range(len(chunks)):
            if i in bad:
                continue
            results.append(
                EmbeddedChunk(
                    chunk_id=chunk_ids[i],
                    text=texts[i],
                    semantic_vector=sem.vectors[i],
                    code_vector=code_vectors.get(i),
                    doc_type=doc_types[i],
                    metadata=metas[i],
                    semantic_identity=semantic_id,
                    code_identity=code_id if i in code_vectors else None,
                )
            )

        return EmbeddingReport(
            chunks=results,
            failures=failures,
            semantic_identity=semantic_id,
            code_identity=code_id,
            dual_vectors=self.dual_vectors_enabled,
            cache_hits=cache_hits,
        )

    def embed_chunks(
        self,
        chunks: list[dict],
        show_progress: bool = True,
        allow_partial: bool = False,
    ) -> list[EmbeddedChunk]:
        """Embed a list of chunk dicts (with 'text', 'metadata' keys).

        Each dict must have:
          - text: str
          - metadata: dict with at least 'chunk_id' and 'doc_type'

        Raises EmbeddingBatchError if any chunk failed, unless
        ``allow_partial=True`` (then only fully embedded chunks are returned;
        use ``embed_chunks_report`` to see the failures).
        """
        report = self.embed_chunks_report(chunks, show_progress=show_progress)
        if report.failures and not allow_partial:
            first = report.failures[0]
            batch = EmbeddingBatch(
                vectors=[],
                identity=report.semantic_identity,  # type: ignore[arg-type]
                failures=[
                    EmbeddingFailure(i, f.error_type, f"{f.chunk_id} [{f.vector_kind}]: {f.message}")
                    for i, f in enumerate(report.failures)
                ],
            )
            raise EmbeddingBatchError(
                f"{len(report.failures)} embedding failure(s) over {len(chunks)} chunks "
                f"(first: {first.chunk_id} [{first.vector_kind}] {first.error_type}: {first.message})",
                batch,
            )
        return report.chunks

    # ------------------------------------------------------------------
    # Cache-aware batch embedding
    # ------------------------------------------------------------------

    def _embed_with_cache(
        self, texts: list[str], model: str, identity: EmbeddingIdentity
    ) -> EmbeddingBatch:
        """Embed texts with the pinned *identity*; cache hits first, misses via client.

        Only validated successes are written to the cache.
        """
        results, miss_indices = self.cache.batch_get(texts, identity)
        failures: list[EmbeddingFailure] = []
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            fresh = self.client.embed_texts(
                miss_texts, model, identity=identity, batch_size=self.batch_size
            )
            if len(fresh.vectors) != len(miss_texts):
                raise InvalidEmbeddingError(
                    f"client returned {len(fresh.vectors)} results for {len(miss_texts)} texts"
                )
            if not fresh.identity.matches(identity):
                raise InvalidEmbeddingError(
                    f"client returned identity {fresh.identity.key}, pinned {identity.key}"
                )
            self.cache.batch_put(miss_texts, identity, fresh.vectors)
            for idx, vec in zip(miss_indices, fresh.vectors):
                results[idx] = vec
            failures = [
                EmbeddingFailure(miss_indices[f.index], f.error_type, f.message)
                for f in fresh.failures
            ]
        return EmbeddingBatch(
            vectors=results,
            identity=identity,
            failures=sorted(failures, key=lambda f: f.index),
            cache_hits=len(texts) - len(miss_indices),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def check_readiness(self) -> dict[str, bool]:
        """Verify an embedding provider is reachable and the models are available."""
        return {
            "ollama_available": self.client.is_available(),
            "prose_model_ready": self.client.has_model(self.prose_model),
            "code_model_ready": self.client.has_model(self.code_model),
        }

    def close(self) -> None:
        self.cache.close()
        self.client.close()
