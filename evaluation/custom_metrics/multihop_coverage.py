from __future__ import annotations

from ._common import aggregate


class MultiHopCoverageMetric:
    """Multi-hop score from cited / linked evidence, not retrieved type diversity.

    ``required_ids`` is the linked evidence cluster the question was built from
    (stored in the dataset). The score is the fraction of those chunks the
    answer cites. When ``links`` (pairs of chunk ids) are supplied, the cited
    required chunks must also form a connected path through them, otherwise the
    score is capped at 0.5.

    Not applicable (None) when the question needs fewer than two chunks.
    """

    name = "multihop_coverage"

    def score_one(
        self,
        cited_ids: list[str],
        required_ids: list[str],
        links: list[tuple[str, str]] | None = None,
    ) -> float | None:
        required = {r for r in required_ids if r}
        if len(required) < 2:
            return None
        hit = required & set(cited_ids or [])
        score = len(hit) / len(required)
        if links and len(hit) >= 2 and not self._connected(hit, links):
            score = min(score, 0.5)
        return score

    @staticmethod
    def _connected(nodes: set[str], links: list[tuple[str, str]]) -> bool:
        adj: dict[str, set[str]] = {}
        for a, b in links:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        start = next(iter(nodes))
        seen, stack = {start}, [start]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return nodes <= seen

    def score(
        self,
        cited: list[list[str]],
        required: list[list[str]],
        links: list[list[tuple[str, str]]] | None = None,
    ) -> float | None:
        links = links or [None] * len(required)  # type: ignore[list-item]
        return aggregate([self.score_one(c, r, l) for c, r, l in zip(cited, required, links)])
