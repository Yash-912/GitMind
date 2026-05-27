from __future__ import annotations

import re


_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


class TemporalAccuracyMetric:
    """Heuristic temporal accuracy metric based on year overlap."""

    name = "temporal_accuracy"

    def score(self, answers: list[str], timestamps: list[list[str]]) -> float:
        if not answers or not timestamps:
            return 0.0
        total = 0
        hits = 0
        for ans, ts_list in zip(answers, timestamps):
            ans_years = set(_YEAR.findall(ans or ""))
            if not ans_years:
                continue
            ctx_years = set()
            for ts in ts_list:
                ctx_years.update(_YEAR.findall(ts or ""))
            total += 1
            if ans_years & ctx_years:
                hits += 1
        return hits / max(1, total)
