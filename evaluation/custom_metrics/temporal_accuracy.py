from __future__ import annotations

from ._common import YEAR, aggregate, content_tokens, sentences


class TemporalAccuracyMetric:
    """Per-claim temporal accuracy.

    Every year in the answer is a claim. A year is fully supported when some
    source sentence (context or ground truth) contains that year *and* shares
    event words with the answer clause that states it. A year that only appears
    in chunk timestamps (no event association) earns half credit; an invented
    year earns zero. The question score is the mean over its years, so one
    unsupported year lowers it.

    Not applicable (None) when the answer states no year.
    """

    name = "temporal_accuracy"

    def score_one(
        self,
        answer: str,
        timestamps: list[str] | None = None,
        context: str = "",
        ground_truth: str = "",
    ) -> float | None:
        claims = [(y, c) for c in sentences(answer) for y in set(YEAR.findall(c))]
        if not claims:
            return None
        sources = sentences(context) + sentences(ground_truth)
        ts_years = {y for ts in (timestamps or []) for y in YEAR.findall(ts or "")}
        total = 0.0
        for year, clause in claims:
            clause_tokens = content_tokens(clause)
            if any(
                year in YEAR.findall(src) and clause_tokens & content_tokens(src)
                for src in sources
            ):
                total += 1.0
            elif year in ts_years:
                total += 0.5
        return total / len(claims)

    def score(
        self,
        answers: list[str],
        timestamps: list[list[str]],
        contexts: list[str] | None = None,
        ground_truths: list[str] | None = None,
    ) -> float | None:
        n = len(answers)
        contexts = contexts or [""] * n
        ground_truths = ground_truths or [""] * n
        ts = list(timestamps) + [[]] * (n - len(timestamps))
        return aggregate(
            [self.score_one(a, t, c, g) for a, t, c, g in zip(answers, ts, contexts, ground_truths)]
        )
