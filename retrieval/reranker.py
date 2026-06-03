from __future__ import annotations

from typing import Iterable

from .types import CandidateChunk


class CrossEncoderReranker:
    """Optional cross-encoder reranker. Falls back to score order."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_size: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.cache_size = cache_size
        self._model = None
        self._cache: dict[tuple[str, str], float] = {}
        self._cache_order: list[tuple[str, str]] = []
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def rerank(self, query: str, candidates: Iterable[CandidateChunk], top_k: int = 12) -> list[CandidateChunk]:
        items = list(candidates)
        if not items:
            return []
        if self._model is None:
            return items[:top_k]

        scores = self._score_with_cache(query, items)
        scored = sorted(zip(items, scores), key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in scored[:top_k]]

    def _score_with_cache(self, query: str, items: list[CandidateChunk]) -> list[float]:
        missing_pairs: list[tuple[str, str]] = []
        missing_texts: list[tuple[str, str]] = []
        scores: list[float] = []

        for c in items:
            key = (query, c.chunk_id)
            if key in self._cache:
                scores.append(self._cache[key])
            else:
                scores.append(None)
                missing_pairs.append(key)
                missing_texts.append((query, c.text))

        if missing_texts:
            new_scores = self._model.predict(missing_texts)
            for key, score in zip(missing_pairs, new_scores):
                self._set_cache(key, float(score))

            # Fill scores list
            it = iter(new_scores)
            scores = [s if s is not None else float(next(it)) for s in scores]

        return scores

    def _set_cache(self, key: tuple[str, str], score: float) -> None:
        if key in self._cache:
            return
        self._cache[key] = score
        self._cache_order.append(key)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
