"""bm25_index.py — BM25 sparse keyword index using bm25s."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """Sparse keyword index built on top of the ``bm25s`` library.

    Stores a mapping from internal index position → chunk_id alongside
    the bm25s index itself.  Persists to disk so it survives restarts.
    """

    def __init__(self, index_dir: str = "data/bm25_index") -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []        # position → chunk_id
        self._texts: list[str] = []      # position → chunk text (for rebuild)
        self._index = None               # bm25s.BM25 instance
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index from scratch over the given corpus."""
        import bm25s  # type: ignore

        self._ids = list(chunk_ids)
        self._texts = list(texts)

        # Tokenize corpus
        corpus_tokens = bm25s.tokenize(texts, stopwords="en")

        # Create and index
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens)

        self._save()

    def add(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Add new documents and rebuild the index.

        BM25s doesn't support incremental add, so we append and rebuild.
        """
        self._ids.extend(chunk_ids)
        self._texts.extend(texts)
        self.build(self._ids, self._texts)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 40) -> list[BM25Result]:
        """Search the index and return ranked results."""
        if self._index is None:
            return []

        import bm25s  # type: ignore

        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self._index.retrieve(
            query_tokens, k=min(limit, len(self._ids))
        )

        hits: list[BM25Result] = []
        for rank in range(results.shape[1]):
            doc_idx = int(results[0, rank])
            score = float(scores[0, rank])
            if doc_idx < len(self._ids):
                hits.append(
                    BM25Result(
                        chunk_id=self._ids[doc_idx],
                        score=score,
                        rank=rank,
                    )
                )
        return hits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._index is not None:
            self._index.save(str(self.index_dir / "bm25_model"))
        with open(self.index_dir / "chunk_ids.json", "w") as f:
            json.dump(self._ids, f)
        with open(self.index_dir / "chunk_texts.json", "w") as f:
            json.dump(self._texts, f)

    def _load_if_exists(self) -> None:
        ids_path = self.index_dir / "chunk_ids.json"
        model_path = self.index_dir / "bm25_model"
        if ids_path.exists() and model_path.exists():
            try:
                import bm25s  # type: ignore

                with open(ids_path, "r") as f:
                    self._ids = json.load(f)
                texts_path = self.index_dir / "chunk_texts.json"
                if texts_path.exists():
                    with open(texts_path, "r") as f:
                        self._texts = json.load(f)
                self._index = bm25s.BM25.load(str(model_path))
            except Exception:
                self._index = None

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._ids)
