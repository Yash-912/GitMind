from __future__ import annotations

from typing import Iterable

from .types import CandidateChunk


class CrossEncoderReranker:
    """Optional cross-encoder reranker. Falls back to score order."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
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

        pairs = [(query, c.text) for c in items]
        scores = self._model.predict(pairs)
        scored = sorted(zip(items, scores), key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in scored[:top_k]]
