from __future__ import annotations

from entities.entity_registry import EntityRegistry


class EntityResolver:
    """Resolve query terms to canonical entities using the registry."""

    def __init__(self, db_path: str) -> None:
        self.registry = EntityRegistry(db_path)

    def resolve(self, terms: list[str]) -> list[str]:
        resolved: list[str] = []
        for term in terms:
            results = self.registry.search(term, limit=5)
            if results:
                resolved.append(results[0].canonical_name)
            else:
                resolved.append(term)
        return list(dict.fromkeys(resolved))

    def close(self) -> None:
        self.registry.close()
