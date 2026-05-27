from __future__ import annotations

import re

from .types import QueryPlan


_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


class QueryDecomposer:
    """Heuristic query decomposition (LLM-backed version is planned)."""

    def decompose(self, query: str) -> QueryPlan:
        years = _YEAR.findall(query)
        time_start = years[0] + "-01-01" if years else None
        time_end = years[-1] + "-12-31" if years else None

        # Simple heuristic: treat quoted phrases as entities
        entities = re.findall(r"\"([^\"]+)\"", query)
        sub_queries = [query]
        return QueryPlan(
            query=query,
            entities=entities,
            time_start=time_start,
            time_end=time_end,
            intent="general",
            sub_queries=sub_queries,
        )
