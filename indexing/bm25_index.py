"""bm25_index.py — BM25 sparse keyword index using bm25s.

Readers and writers
-------------------
Constructing a ``BM25Index`` never writes: it neither creates ``index_dir`` nor
touches files, so it can be opened on a published generation's ``bm25_dir``.
Only ``build()``/``add()`` write, and every file is written to a temporary
name first and then renamed into place, so a concurrent reader never sees a
half-written file.  Builds are meant to target a *new* generation directory
(see ``indexing/generation.py``), never the directory a reader is using.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    rank: int


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f)
    os.replace(tmp, path)


class BM25Index:
    """Sparse keyword index built on top of the ``bm25s`` library.

    Stores a mapping from internal index position → chunk_id alongside
    the bm25s index itself.  Persists to disk so it survives restarts.
    """

    def __init__(self, index_dir: str = "data/bm25_index") -> None:
        self.index_dir = Path(index_dir)
        self._ids: list[str] = []        # position → chunk_id
        self._texts: list[str] = []      # position → chunk text (for rebuild)
        self._index = None               # bm25s.BM25 instance
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index from scratch over the given corpus."""
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) differ in length"
            )
        if not chunk_ids or not texts:
            self._ids = []
            self._texts = []
            self._index = None
            return
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

    def search(
        self,
        query: str,
        limit: int = 40,
        allowed_ids: "Iterable[str] | None" = None,
    ) -> list[BM25Result]:
        """Search the index and return ranked results.

        With ``allowed_ids`` the restriction is applied *before* truncating to
        ``limit`` (filter-then-rank): documents outside the set can never
        occupy a result slot, so up to ``limit`` allowed hits are returned.
        """
        if self._index is None or not self._ids or limit <= 0:
            return []

        import bm25s  # type: ignore
        import numpy as np

        query_tokens = bm25s.tokenize([query], stopwords="en")

        allowed: set[str] | None = None
        kwargs: dict = {}
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            mask = np.array([1.0 if cid in allowed else 0.0 for cid in self._ids])
            n_allowed = int(mask.sum())
            if n_allowed == 0:
                return []
            k = min(limit, n_allowed)
            kwargs["weight_mask"] = mask
        else:
            k = min(limit, len(self._ids))

        results, scores = self._index.retrieve(query_tokens, k=k, **kwargs)

        hits: list[BM25Result] = []
        for pos in range(results.shape[1]):
            doc_idx = int(results[0, pos])
            if doc_idx >= len(self._ids):
                continue
            chunk_id = self._ids[doc_idx]
            if allowed is not None and chunk_id not in allowed:
                continue
            hits.append(
                BM25Result(chunk_id=chunk_id, score=float(scores[0, pos]), rank=len(hits))
            )
        return hits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            # bm25s saves a directory; write it aside, then swap it in.
            tmp_model = self.index_dir / f".bm25_model.{uuid.uuid4().hex}.tmp"
            self._index.save(str(tmp_model))
            final = self.index_dir / "bm25_model"
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp_model, final)
        _atomic_json(self.index_dir / "chunk_ids.json", self._ids)
        _atomic_json(self.index_dir / "chunk_texts.json", self._texts)

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
