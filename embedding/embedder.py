"""embedder.py — Dual embedding orchestrator with caching and batching."""
from __future__ import annotations

from dataclasses import dataclass

from .models import OllamaEmbeddingClient, PROSE_MODEL, CODE_MODEL, PROSE_DIM, CODE_DIM
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


# Doc types that should get code-specific embeddings
_CODE_DOC_TYPES = {"diff", "code"}


class DualEmbedder:
    """Embed chunks with nomic-embed-text (prose) and nomic-embed-code (diffs).

    Every chunk gets a semantic embedding.  Chunks with doc_type in
    {"diff", "code"} also get a code-specific embedding.

    Uses EmbeddingCache to skip already-embedded chunks on re-ingestion.
    """

    def __init__(
        self,
        client: OllamaEmbeddingClient | None = None,
        cache: EmbeddingCache | None = None,
        prose_model: str = PROSE_MODEL,
        code_model: str = CODE_MODEL,
        batch_size: int = 32,
    ) -> None:
        self.client = client or OllamaEmbeddingClient()
        self.cache = cache or EmbeddingCache()
        self.prose_model = prose_model
        self.code_model = code_model
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[dict],
        show_progress: bool = True,
    ) -> list[EmbeddedChunk]:
        """Embed a list of chunk dicts (with 'text', 'metadata' keys).

        Each dict must have:
          - text: str
          - metadata: dict with at least 'chunk_id' and 'doc_type'

        Returns EmbeddedChunk objects with vectors populated.
        """
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        chunk_ids = [m["chunk_id"] for m in metas]
        doc_types = [m.get("doc_type", "") for m in metas]

        # ---- Semantic embeddings (all chunks) ----
        if show_progress:
            print(f"  Embedding {len(texts)} chunks with {self.prose_model}...")
        semantic_vectors = self._embed_with_cache(texts, self.prose_model)

        # ---- Code embeddings (only diff/code chunks) ----
        code_indices = [i for i, dt in enumerate(doc_types) if dt in _CODE_DOC_TYPES]
        code_vectors: dict[int, list[float]] = {}
        if code_indices:
            code_texts = [texts[i] for i in code_indices]
            if show_progress:
                print(f"  Embedding {len(code_texts)} code chunks with {self.code_model}...")
            code_vecs = self._embed_with_cache(code_texts, self.code_model)
            for idx, vec in zip(code_indices, code_vecs):
                code_vectors[idx] = vec

        # ---- Assemble results ----
        results: list[EmbeddedChunk] = []
        for i in range(len(chunks)):
            results.append(
                EmbeddedChunk(
                    chunk_id=chunk_ids[i],
                    text=texts[i],
                    semantic_vector=semantic_vectors[i],
                    code_vector=code_vectors.get(i),
                    doc_type=doc_types[i],
                    metadata=metas[i],
                )
            )

        return results

    # ------------------------------------------------------------------
    # Cache-aware batch embedding
    # ------------------------------------------------------------------

    def _embed_with_cache(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        """Embed texts, using cache for hits and Ollama for misses."""
        cached_results, miss_indices = self.cache.batch_get(texts, model)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            new_vectors = self.client.embed_batch(
                miss_texts, model=model, batch_size=self.batch_size
            )
            # Store in cache and fill results
            self.cache.batch_put(miss_texts, model, new_vectors)
            for idx, vec in zip(miss_indices, new_vectors):
                cached_results[idx] = vec

        # At this point all should be non-None
        return [v for v in cached_results]  # type: ignore

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def check_readiness(self) -> dict[str, bool]:
        """Verify Ollama is running and both models are available."""
        return {
            "ollama_available": self.client.is_available(),
            "prose_model_ready": self.client.has_model(self.prose_model),
            "code_model_ready": self.client.has_model(self.code_model),
        }

    def close(self) -> None:
        self.cache.close()
        self.client.close()
