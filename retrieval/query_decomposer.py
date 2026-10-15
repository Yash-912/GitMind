from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

from generation.llm_client import LLMClient
from .types import QueryPlan

log = logging.getLogger(__name__)

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

KNOWN_INTENTS = ("ownership", "risk", "decision", "why", "temporal", "general")

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ownership", re.compile(
        r"\b(who\s+(owns|maintains|wrote|authored|is\s+responsible)|owner(s|ship)?|maintainers?"
        r"|authored\s+by|written\s+by|blame)\b", re.I)),
    ("risk", re.compile(
        r"\b(risks?|risky|fragile|bug-?prone|churn|regress\w*|unstable|hotspots?)\b", re.I)),
    ("decision", re.compile(
        r"\b(decid\w*|decisions?|chose|choose|trade-?offs?|rationale|adopt\w*)\b", re.I)),
    ("why", re.compile(r"\b(why|reasons?|cause[sd]?|led\s+to|because)\b", re.I)),
    ("temporal", re.compile(
        r"\b(when|before|after|since|timeline|history|over\s+time|first|last|between)\b|\b(19|20)\d{2}\b",
        re.I)),
]

# Conservative: a person only becomes an author filter for queries like these.
_AUTHORSHIP = re.compile(
    r"\b(who\s+(wrote|authored|committed|owns|maintains)|written\s+by|authored\s+by"
    r"|commits?\s+(by|from)|author\s+of|contributions?\s+(by|from))\b",
    re.I,
)

_SPLIT = re.compile(
    r"\?\s+(?=\S)|;\s+|\s+and\s+(?:also\s+)?(?=(?:what|who|why|when|how|which|where)\b)", re.I
)


def classify_intent(query: str) -> str:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "general"


def is_authorship_query(query: str) -> bool:
    return bool(_AUTHORSHIP.search(query))


def split_sub_queries(query: str) -> list[str]:
    """Split a compound question into independent parts (>= 3 words each)."""
    parts = [p.strip(" ?;,.") for p in _SPLIT.split(query)]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts if len(parts) > 1 else []


class _PlanSchema(BaseModel):
    entities: list[str] = []
    time_start: str | None = None
    time_end: str | None = None
    intent: str = "general"
    sub_queries: list[str] = []


class QueryDecomposer:
    """LLM-assisted query decomposition with heuristic fallback.

    The plan's ``sub_queries`` are the independent parts of a compound
    question; the original query is always retrieved too (by the service).
    LLM failure is not silent: it is recorded in ``plan.warnings``.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def decompose(self, query: str) -> QueryPlan:
        plan = self._heuristic(query)
        llm_plan, error = self._llm_plan(query)
        if llm_plan is None:
            if error:
                plan.warnings.append(f"query decomposer LLM unavailable ({error}); used heuristic plan")
            return plan
        if llm_plan.intent not in KNOWN_INTENTS or llm_plan.intent == "general":
            llm_plan.intent = plan.intent
        if len(llm_plan.sub_queries) <= 1:
            llm_plan.sub_queries = plan.sub_queries
        llm_plan.time_start = llm_plan.time_start or plan.time_start
        llm_plan.time_end = llm_plan.time_end or plan.time_end
        llm_plan.authorship = plan.authorship
        return llm_plan

    def _heuristic(self, query: str) -> QueryPlan:
        years = _YEAR.findall(query)
        time_start = years[0] + "-01-01" if years else None
        time_end = years[-1] + "-12-31" if years else None

        entities = re.findall(r"\"([^\"]+)\"", query)
        return QueryPlan(
            query=query,
            entities=entities,
            time_start=time_start,
            time_end=time_end,
            intent=classify_intent(query),
            sub_queries=split_sub_queries(query) or [query],
            authorship=is_authorship_query(query),
        )

    def _llm_plan(self, query: str) -> tuple[QueryPlan | None, str | None]:
        prompt = (
            "Extract a structured query plan as JSON with keys: "
            "entities (list), time_start (ISO date or null), "
            "time_end (ISO date or null), intent (one of "
            f"{', '.join(KNOWN_INTENTS)}), sub_queries (list of independent questions).\n\n"
            f"Query: {query}\n\nReturn only JSON."
        )
        try:
            resp = self.llm.generate(prompt).text
            data = json.loads(resp)
            plan = _PlanSchema.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - surfaced via plan.warnings
            log.warning("query decomposition LLM failed: %s", exc)
            return None, f"{type(exc).__name__}: {exc}"
        return (
            QueryPlan(
                query=query,
                entities=list(plan.entities or []),
                time_start=plan.time_start,
                time_end=plan.time_end,
                intent=(plan.intent or "general").lower(),
                sub_queries=plan.sub_queries or [query],
            ),
            None,
        )

    def close(self) -> None:
        self.llm.close()
