from __future__ import annotations

import json
import re

from entities.entity_registry import EntityRegistry
from generation.llm_client import LLMClient


class EntityResolver:
    """Resolve query terms to canonical entities using the registry."""

    def __init__(self, db_path: str, llm: LLMClient | None = None) -> None:
        self.registry = EntityRegistry(db_path)
        self.llm = llm or LLMClient()

    def resolve(self, terms: list[str]) -> list[str]:
        resolved: list[str] = []
        for term in terms:
            results = self.registry.search(term, limit=5)
            if results:
                resolved.append(results[0].canonical_name)
            else:
                resolved.append(term)
        return list(dict.fromkeys(resolved))

    def resolve_from_query(self, query: str) -> list[str]:
        extracted = self._llm_extract_entities(query)
        if not extracted:
            extracted = re.findall(r"\"([^\"]+)\"", query)
        if not extracted:
            extracted = query.split()
        return self.resolve(extracted)

    def _llm_extract_entities(self, query: str) -> list[str]:
        prompt = (
            "Extract a JSON list of the key entities from the query. "
            "Return only JSON.\n\n"
            f"Query: {query}"
        )
        try:
            resp = self.llm.generate(prompt).text
            data = json.loads(resp)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
        return []

    def close(self) -> None:
        self.llm.close()
        self.registry.close()
