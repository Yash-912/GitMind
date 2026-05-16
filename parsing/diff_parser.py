"""diff_parser.py — Parse unified diff text into structured hunks using unidiff."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiffHunk:
    """One parsed hunk from a unified diff."""

    file_path: str          # effective (new) file path
    old_path: str           # a/ path
    new_path: str           # b/ path
    old_start: int          # original start line
    old_count: int
    new_start: int          # new start line
    new_count: int
    header: str             # @@ ... @@ line
    lines: list[str] = field(default_factory=list)   # raw diff lines (+/-/ )
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)

    @property
    def added_lines(self) -> list[str]:
        return [l[1:] for l in self.lines if l.startswith("+")]

    @property
    def removed_lines(self) -> list[str]:
        return [l[1:] for l in self.lines if l.startswith("-")]

    @property
    def text(self) -> str:
        """Full hunk text including header."""
        return self.header + "\n" + "\n".join(self.lines)

    def token_estimate(self, chars_per_token: int = 4) -> int:
        return len(self.text) // chars_per_token


class DiffParser:
    """Parse raw unified-diff text into a list of DiffHunk objects.

    Uses the ``unidiff`` library when available, with a lightweight
    fallback regex parser so the system works even without the dependency.
    """

    def parse(self, diff_text: str) -> list[DiffHunk]:
        if not diff_text or not diff_text.strip():
            return []
        try:
            return self._parse_with_unidiff(diff_text)
        except Exception:
            return self._parse_fallback(diff_text)

    # ------------------------------------------------------------------
    # Primary parser (requires `unidiff`)
    # ------------------------------------------------------------------

    def _parse_with_unidiff(self, diff_text: str) -> list[DiffHunk]:
        import unidiff  # type: ignore

        patch_set = unidiff.PatchSet(diff_text)
        hunks: list[DiffHunk] = []
        for patched_file in patch_set:
            old_path = patched_file.source_file or ""
            new_path = patched_file.target_file or ""
            file_path = new_path.lstrip("b/") if new_path.startswith("b/") else new_path
            for hunk in patched_file:
                lines = [str(l) for l in hunk]
                hunks.append(
                    DiffHunk(
                        file_path=file_path,
                        old_path=old_path,
                        new_path=new_path,
                        old_start=hunk.source_start,
                        old_count=hunk.source_length,
                        new_start=hunk.target_start,
                        new_count=hunk.target_length,
                        header=str(hunk.section_header) if hunk.section_header else f"@@ -{hunk.source_start},{hunk.source_length} +{hunk.target_start},{hunk.target_length} @@",
                        lines=lines,
                    )
                )
        return hunks

    # ------------------------------------------------------------------
    # Fallback parser (stdlib only)
    # ------------------------------------------------------------------

    def _parse_fallback(self, diff_text: str) -> list[DiffHunk]:
        import re

        hunks: list[DiffHunk] = []
        current_file_old = ""
        current_file_new = ""
        current_hunk: DiffHunk | None = None
        hunk_pattern = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)"
        )

        for line in diff_text.splitlines():
            if line.startswith("--- "):
                current_file_old = line[4:].strip()
                if current_hunk is not None:
                    hunks.append(current_hunk)
                    current_hunk = None
            elif line.startswith("+++ "):
                current_file_new = line[4:].strip()
            elif m := hunk_pattern.match(line):
                if current_hunk is not None:
                    hunks.append(current_hunk)
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) else 1
                file_path = current_file_new.lstrip("b/") if current_file_new.startswith("b/") else current_file_new
                current_hunk = DiffHunk(
                    file_path=file_path,
                    old_path=current_file_old,
                    new_path=current_file_new,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=line,
                    lines=[],
                )
            elif current_hunk is not None and (
                line.startswith("+")
                or line.startswith("-")
                or line.startswith(" ")
            ):
                current_hunk.lines.append(line)

        if current_hunk is not None:
            hunks.append(current_hunk)

        return hunks
