from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Iterable

from .types import CandidateChunk

log = logging.getLogger(__name__)

GRAPH_SOURCE = "graph"


class CrossEncoderReranker:
    """Cross-encoder reranker, loaded once for the lifetime of its owner.

    If the model cannot be loaded, ``available`` is False, ``load_error`` holds
    the reason, a warning is logged, and ``rerank`` falls back to the incoming
    (retrieval-score) order. The cross-encoder score is stored in
    ``CandidateChunk.rerank_score``; the retrieval ``score`` is never touched.
    Either way, ``graph_slots`` result slots are reserved for graph evidence.

    ``model`` lets tests inject an object with ``predict(list[(query, text)])``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_size: int = 2048,
        model: Any = None,
        graph_slots: int = 3,
    ) -> None:
        self.model_name = model_name
        self.cache_size = cache_size
        self.graph_slots = graph_slots
        self.load_error: str | None = None
        self._model = model
        self._cache: dict[tuple[str, str], float] = {}
        self._cache_order: list[tuple[str, str]] = []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(model_name)
            except Exception as exc:  # noqa: BLE001 - recorded + logged, not swallowed
                self._model = None
                self.load_error = f"{type(exc).__name__}: {exc}"
                log.warning("reranker unavailable (%s): %s", model_name, self.load_error)

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: Iterable[CandidateChunk],
        top_k: int = 12,
    ) -> list[CandidateChunk]:
        items = list(candidates)
        if not items:
            return []
        if self._model is None:
            ranked = list(items)  # retrieval order, rerank_score stays None
        else:
            scores = self._score_with_cache(query, items)
            order = sorted(range(len(items)), key=lambda i: float(scores[i]), reverse=True)
            ranked = [replace(items[i], rerank_score=float(scores[i])) for i in order]
        return self._select(ranked, top_k)

    def _select(self, ranked: list[CandidateChunk], top_k: int) -> list[CandidateChunk]:
        """Top-k in rank order, guaranteeing graph evidence keeps its reserved slots."""
        chosen = ranked[:top_k]
        graph_all = [c for c in ranked if c.source == GRAPH_SOURCE]
        want = min(self.graph_slots, len(graph_all), top_k)
        have = sum(1 for c in chosen if c.source == GRAPH_SOURCE)
        if have >= want:
            return chosen
        extra = [c for c in graph_all if c not in chosen][: want - have]
        # Evict the lowest-ranked non-graph candidates to make room.
        drop = 0
        keep: list[CandidateChunk] = []
        for c in reversed(chosen):
            if drop < len(extra) and c.source != GRAPH_SOURCE:
                drop += 1
                continue
            keep.append(c)
        keep.reverse()
        selected = set(id(c) for c in keep + extra)
        return [c for c in ranked if id(c) in selected]

    def _score_with_cache(self, query: str, items: list[CandidateChunk]) -> list[float]:
        missing_indices: list[int] = []
        missing_texts: list[tuple[str, str]] = []
        scores: list[float] = [0.0] * len(items)

        for i, c in enumerate(items):
            key = (query, c.chunk_id)
            if key in self._cache:
                scores[i] = self._cache[key]
            else:
                missing_indices.append(i)
                missing_texts.append((query, c.text))

        if missing_texts:
            new_scores = list(self._model.predict(missing_texts))
            for list_idx, item_idx in enumerate(missing_indices):
                key = (query, items[item_idx].chunk_id)
                score = float(new_scores[list_idx])
                self._set_cache(key, score)
                scores[item_idx] = score

        return scores

    def _set_cache(self, key: tuple[str, str], score: float) -> None:
        if key in self._cache:
            return
        self._cache[key] = score
        self._cache_order.append(key)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
