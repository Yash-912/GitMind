"""tagging.py — Canonical module / entity tags for chunks (audit F05).

* ``module_tags``: path-derived modules of the files a chunk touches, plus
  module and symbol entities found in the document text.
* ``entity_tags``: people and technologies.

Tags keep the case of the source path / canonical entity name.
"""
from __future__ import annotations

import re
from typing import Iterable

from .entity_registry import (
    KIND_MODULE,
    KIND_PERSON,
    KIND_SYMBOL,
    KIND_TECHNOLOGY,
    EntityRegistry,
)

# Container directories that say nothing about the module.
GENERIC_ROOTS = frozenset(
    {"src", "lib", "app", "source", "sources", "pkg", "packages", "internal", "cmd"}
)

_PATH_IN_TEXT = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)+[\w.-]+\.\w{1,6})(?![\w/])")


def module_tags_for_paths(paths: Iterable[str], depth: int = 2) -> list[str]:
    """Module tags for file paths: first non-generic directory segment(s).

    ``src/auth/jwt/verify.py`` -> ``["auth", "auth/jwt"]``; a root-level file
    ``setup.py`` -> ``["setup"]``. Order is stable and duplicates removed.
    """
    tags: dict[str, None] = {}
    for raw in paths:
        parts = [p for p in raw.replace("\\", "/").split("/") if p and p != "."]
        if not parts:
            continue
        dirs = parts[:-1]
        while dirs and dirs[0].lower() in GENERIC_ROOTS:
            dirs = dirs[1:]
        if not dirs:
            stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1][1:] else parts[-1]
            if stem:
                tags[stem] = None
            continue
        for n in range(1, min(depth, len(dirs)) + 1):
            tags["/".join(dirs[:n])] = None
    return list(tags)


def paths_in_text(text: str) -> list[str]:
    """File-like paths (containing a directory part) mentioned in free text."""
    return list(dict.fromkeys(_PATH_IN_TEXT.findall(text or "")))


def resolve_entities(
    extractor,
    registry: EntityRegistry,
    text: str,
    doc_id: str,
    doc_type: str,
    author: str = "",
) -> list[tuple[str, str]]:
    """Extract entities from ``text``, resolve them in ``registry`` and record
    (repo, doc_type, doc_id) document references. Returns (canonical, kind)."""
    resolved: list[tuple[str, str]] = []
    for e in extractor.extract(text, doc_id=doc_id, doc_type=doc_type):
        canonical = registry.resolve(e.text, e.label.lower())
        registry.add_doc_reference(canonical, doc_id, doc_type=doc_type, kind=e.label.lower())
        rec = registry.get(canonical, e.label.lower())
        if rec is not None:
            resolved.append((canonical, rec.kind))
    if author:
        canonical = registry.resolve(author, "person")
        registry.add_doc_reference(canonical, doc_id, doc_type=doc_type, kind="person")
        resolved.append((canonical, KIND_PERSON))
    return resolved


def split_tags(resolved: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """(module_tags, entity_tags) from resolved (canonical, kind) entities."""
    modules: dict[str, None] = {}
    entities: dict[str, None] = {}
    for name, kind in resolved:
        if kind in (KIND_MODULE, KIND_SYMBOL):
            modules[name] = None
        elif kind in (KIND_PERSON, KIND_TECHNOLOGY):
            entities[name] = None
    return list(modules), list(entities)


def stamp_tags(chunks, module_tags: Iterable[str] = (), entity_tags: Iterable[str] = ()) -> None:
    """Merge tags into every chunk's metadata, keeping order and uniqueness.

    Each chunk also gets module tags derived from its own ``file_paths``.
    """
    for c in chunks:
        m = c.metadata
        own = module_tags_for_paths(m.file_paths)
        m.module_tags = list(dict.fromkeys([*m.module_tags, *own, *module_tags]))
        m.entity_tags = list(dict.fromkeys([*m.entity_tags, *entity_tags]))
