from __future__ import annotations

import json
import re
from dataclasses import dataclass

from entities.entity_registry import EntityRegistry
from generation.llm_client import LLMClient


@dataclass(frozen=True)
class ResolvedEntity:
    name: str      # canonical name (or the raw term when unknown)
    kind: str      # module | symbol | person | technology | concept | unknown
    surface: str   # the form used in the query


class EntityResolver:
    """Resolve query terms to canonical entities using the registry."""

    def __init__(self, db_path: str, llm: LLMClient | None = None, repo: str = "") -> None:
        # The registry is repo-scoped: without the active repo it sees nothing.
        self.repo = repo
        self.registry = EntityRegistry(db_path, repo=repo)
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

    # ---- typed resolution (F05) -------------------------------------- #

    def resolve_typed(self, terms: list[str]) -> list["ResolvedEntity"]:
        """Resolve terms to entities *with their kind*.

        Only an exact (case-insensitive) match on a canonical name or alias
        counts: the registry's prefix search would otherwise turn ordinary words
        into module filters. Anything unmatched has kind ``unknown`` and must
        never be used as a filter.
        """
        out: list[ResolvedEntity] = []
        for term in dict.fromkeys(t.strip() for t in terms if t and t.strip()):
            folded = term.casefold()
            match = None
            for rec in self.registry.search(term, limit=10):
                names = {rec.canonical_name.casefold(), *(a.casefold() for a in rec.aliases)}
                if folded in names:
                    match = rec
                    break
            if match is None:
                out.append(ResolvedEntity(name=term, kind="unknown", surface=term))
                continue
            out.append(ResolvedEntity(match.canonical_name, match.kind, term))
        return out

    def resolve_from_query_typed(self, query: str) -> list["ResolvedEntity"]:
        extracted = self._llm_extract_entities(query)
        if not extracted:
            extracted = re.findall(r"\"([^\"]+)\"", query)
        if not extracted:
            extracted = query.split()
        return self.resolve_typed(extracted)

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
