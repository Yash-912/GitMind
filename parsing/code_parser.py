"""code_parser.py — Extract code entities (functions, classes, imports) using tree-sitter."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CodeEntity:
    """A top-level code entity extracted from a source file."""

    entity_type: str        # "function" | "class" | "import" | "method"
    name: str
    start_line: int
    end_line: int
    file_path: str
    language: str
    text: str               # raw source of the entity
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None


# Map common file extensions → tree-sitter language name
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
}


def _extension(file_path: str) -> str:
    import os

    _, ext = os.path.splitext(file_path)
    return ext.lower()


class CodeParser:
    """Extract code entities from source text.

    Attempts to use tree-sitter when the parser + language binding are
    installed.  Falls back to a regex-based heuristic for Python and
    JavaScript so the system is always functional.
    """

    def parse(self, source: str, file_path: str = "") -> list[CodeEntity]:
        """Return a list of top-level CodeEntity objects found in *source*."""
        lang = _EXT_TO_LANG.get(_extension(file_path), "unknown")
        try:
            return self._parse_with_tree_sitter(source, file_path, lang)
        except Exception:
            return self._parse_fallback(source, file_path, lang)

    # ------------------------------------------------------------------
    # tree-sitter implementation
    # ------------------------------------------------------------------

    def _parse_with_tree_sitter(
        self, source: str, file_path: str, lang: str
    ) -> list[CodeEntity]:
        """Use tree-sitter Python bindings.  Raises on import or parse failure."""
        from tree_sitter import Language, Parser  # type: ignore
        import tree_sitter_python as tspython  # type: ignore

        # Only Python is bundled; extend as needed.
        lang_map = {"python": tspython}
        if lang not in lang_map:
            raise ValueError(f"No tree-sitter binding for {lang}")

        PY_LANGUAGE = Language(lang_map[lang].language())
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(source.encode("utf-8", errors="replace"))
        root = tree.root_node

        entities: list[CodeEntity] = []
        lines = source.splitlines()

        def visit(node, parent_class: str | None = None) -> None:
            if node.type in ("function_definition", "async_function_definition"):
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode() if name_node else "?"
                start = node.start_point[0]
                end = node.end_point[0]
                entity_type = "method" if parent_class else "function"
                entities.append(
                    CodeEntity(
                        entity_type=entity_type,
                        name=name,
                        start_line=start + 1,
                        end_line=end + 1,
                        file_path=file_path,
                        language=lang,
                        text="\n".join(lines[start : end + 1]),
                        parent_class=parent_class,
                    )
                )
                # Recurse into class body methods
                for child in node.children:
                    visit(child, parent_class=parent_class)
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode() if name_node else "?"
                start = node.start_point[0]
                end = node.end_point[0]
                entities.append(
                    CodeEntity(
                        entity_type="class",
                        name=name,
                        start_line=start + 1,
                        end_line=end + 1,
                        file_path=file_path,
                        language=lang,
                        text="\n".join(lines[start : end + 1]),
                    )
                )
                for child in node.children:
                    visit(child, parent_class=name)
            elif node.type in ("import_statement", "import_from_statement"):
                start = node.start_point[0]
                entities.append(
                    CodeEntity(
                        entity_type="import",
                        name=node.text.decode(errors="replace").split("\n")[0],
                        start_line=start + 1,
                        end_line=start + 1,
                        file_path=file_path,
                        language=lang,
                        text=node.text.decode(errors="replace"),
                    )
                )
            else:
                for child in node.children:
                    visit(child, parent_class=parent_class)

        visit(root)
        return entities

    # ------------------------------------------------------------------
    # Regex fallback (Python + JavaScript heuristics)
    # ------------------------------------------------------------------

    _PY_FUNC = re.compile(r"^( *)(?:async )?def (\w+)\s*\(", re.MULTILINE)
    _PY_CLASS = re.compile(r"^( *)class (\w+)[\s:(]", re.MULTILINE)
    _JS_FUNC = re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE
    )
    _JS_ARROW = re.compile(
        r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE
    )

    def _parse_fallback(
        self, source: str, file_path: str, lang: str
    ) -> list[CodeEntity]:
        entities: list[CodeEntity] = []
        lines = source.splitlines()
        line_starts = [0]
        pos = 0
        for line in lines:
            pos += len(line) + 1
            line_starts.append(pos)

        def line_num(char_pos: int) -> int:
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= char_pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        if lang == "python":
            for m in self._PY_FUNC.finditer(source):
                ln = line_num(m.start())
                indent = len(m.group(1))
                entities.append(
                    CodeEntity(
                        entity_type="function",
                        name=m.group(2),
                        start_line=ln,
                        end_line=ln,
                        file_path=file_path,
                        language=lang,
                        text=lines[ln - 1] if ln <= len(lines) else "",
                    )
                )
            for m in self._PY_CLASS.finditer(source):
                ln = line_num(m.start())
                entities.append(
                    CodeEntity(
                        entity_type="class",
                        name=m.group(2),
                        start_line=ln,
                        end_line=ln,
                        file_path=file_path,
                        language=lang,
                        text=lines[ln - 1] if ln <= len(lines) else "",
                    )
                )
        elif lang in ("javascript", "typescript"):
            for m in self._JS_FUNC.finditer(source):
                ln = line_num(m.start())
                entities.append(
                    CodeEntity(
                        entity_type="function",
                        name=m.group(1),
                        start_line=ln,
                        end_line=ln,
                        file_path=file_path,
                        language=lang,
                        text=lines[ln - 1] if ln <= len(lines) else "",
                    )
                )
            for m in self._JS_ARROW.finditer(source):
                ln = line_num(m.start())
                entities.append(
                    CodeEntity(
                        entity_type="function",
                        name=m.group(1),
                        start_line=ln,
                        end_line=ln,
                        file_path=file_path,
                        language=lang,
                        text=lines[ln - 1] if ln <= len(lines) else "",
                    )
                )

        return entities
