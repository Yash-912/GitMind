from __future__ import annotations

class MultiHopCoverageMetric:
    """Heuristic multi-hop coverage metric."""

    name = "multihop_coverage"

    def score(self, doc_types: list[list[str]]) -> float:
        if not doc_types:
            return 0.0
        multi = sum(1 for d in doc_types if len(set([x for x in d if x])) > 1)
        return multi / max(1, len(doc_types))
