from __future__ import annotations

import re


_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9_]{2,}|[A-Z]{2,})\b")


class EntityConsistencyMetric:
    """Heuristic entity consistency metric using entity_tags and answer text."""

    name = "entity_consistency"

    def score(self, answers: list[str], entity_tags: list[list[list[str]]]) -> float:
        if not answers or not entity_tags:
            return 0.0
        total = 0
        hits = 0
        for ans, tags in zip(answers, entity_tags):
            ans_entities = {m.group(1).lower() for m in _ENTITY.finditer(ans or "")}
            if not ans_entities:
                continue
            ctx_entities = {t.lower() for group in tags for t in group if t}
            total += 1
            if ans_entities & ctx_entities:
                hits += 1
        return hits / max(1, total)
