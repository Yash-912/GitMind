from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from generation.llm_client import LLMClient
from .types import QueryPlan


_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


class _PlanSchema(BaseModel):
    entities: list[str] = []
    time_start: str | None = None
    time_end: str | None = None
    intent: str = "general"
    sub_queries: list[str] = []


class QueryDecomposer:
    """LLM-assisted query decomposition with heuristic fallback."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def decompose(self, query: str) -> QueryPlan:
        plan = self._heuristic(query)
        llm_plan = self._llm_plan(query)
        return llm_plan or plan

    def _heuristic(self, query: str) -> QueryPlan:
        years = _YEAR.findall(query)
        time_start = years[0] + "-01-01" if years else None
        time_end = years[-1] + "-12-31" if years else None

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

    def _llm_plan(self, query: str) -> QueryPlan | None:
        prompt = (
            "Extract a structured query plan as JSON with keys: "
            "entities (list), time_start (ISO date or null), "
            "time_end (ISO date or null), intent (string), "
            "sub_queries (list).\n\n"
            f"Query: {query}\n\nReturn only JSON."
        )
        try:
            resp = self.llm.generate(prompt).text
            data = json.loads(resp)
            plan = _PlanSchema.model_validate(data)
            sub_queries = plan.sub_queries or [query]
            return QueryPlan(
                query=query,
                entities=list(plan.entities or []),
                time_start=plan.time_start,
                time_end=plan.time_end,
                intent=plan.intent or "general",
                sub_queries=sub_queries,
            )
        except Exception:
            return None

    def close(self) -> None:
        self.llm.close()
