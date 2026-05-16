"""changelog_chunker.py — Chunk CHANGELOG.md / release notes per version section."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedRelease

# PRD §7.5: split per version section, e.g. "## v2.3.0 — 2022-04-10"
# Also handle: "# Changelog", "## [2.3.0] – 2022-04-10", "## 2.3.0 (2022-04-10)"
_VERSION_HEADING = re.compile(
    r"^(#{1,3} .*(?:v?\d+\.\d+[\.\d]*|unreleased).*)",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ChangelogSection:
    heading: str
    body: str
    version: str
    date: datetime | None


def _parse_date_from_heading(heading: str) -> datetime | None:
    date_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\w+ \d{1,2},? \d{4})"
    )
    m = date_pattern.search(heading)
    if not m:
        return None
    raw = m.group(1).replace("/", "-")
    for fmt in ("%Y-%m-%d", "%B %d %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_version_from_heading(heading: str) -> str:
    version_pattern = re.compile(r"v?(\d+\.\d+[\.\d]*)")
    m = version_pattern.search(heading)
    return m.group(0) if m else heading.strip("# ").split()[0]


def parse_changelog(text: str) -> list[ChangelogSection]:
    """Split a full CHANGELOG text into version sections."""
    splits = _VERSION_HEADING.split(text)
    sections: list[ChangelogSection] = []
    # splits alternates: [preamble, heading, body, heading, body, ...]
    i = 0
    while i < len(splits):
        token = splits[i]
        if _VERSION_HEADING.match(token):
            heading = token.strip()
            body = splits[i + 1].strip() if i + 1 < len(splits) else ""
            sections.append(
                ChangelogSection(
                    heading=heading,
                    body=body,
                    version=_parse_version_from_heading(heading),
                    date=_parse_date_from_heading(heading),
                )
            )
            i += 2
        else:
            i += 1
    return sections


class ChangelogChunker(BaseChunker):
    """One chunk per version section in a CHANGELOG or release notes body."""

    def __init__(self, max_tokens: int = 600) -> None:
        super().__init__(max_tokens=max_tokens)

    def chunk(self, parsed_doc: ParsedRelease, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        release = parsed_doc
        body = release.body_clean or ""

        # Try to split into sub-sections (useful for aggregated CHANGELOG files)
        sections = parse_changelog(body)
        if not sections:
            # Single release note: treat as one chunk
            text = f"[Release {release.tag}]\n\n{body}"
            meta = ChunkMetadata(
                doc_type="release",
                doc_id=release.tag,
                timestamp=release.timestamp or datetime.utcnow(),
                repo=repo,
                graph_node_id=f"release_{release.tag}",
                total_chunks=1,
                chunk_index=0,
            )
            return [Chunk(text=text[:self.max_chars], metadata=meta)]

        chunks: list[Chunk] = []
        for idx, section in enumerate(sections):
            text = f"[Release {section.version}]\n{section.heading}\n\n{section.body}"
            ts = section.date or release.timestamp or datetime.utcnow()
            meta = ChunkMetadata(
                doc_type="release",
                doc_id=section.version,
                timestamp=ts,
                repo=repo,
                graph_node_id=f"release_{section.version}",
                chunk_index=idx,
                total_chunks=len(sections),
            )
            if len(text) > self.max_chars:
                for sub_idx, sub_text in enumerate(self._split_by_chars(text)):
                    import copy
                    sub_meta = copy.deepcopy(meta)
                    chunks.append(Chunk(text=sub_text, metadata=sub_meta))
            else:
                chunks.append(Chunk(text=text, metadata=meta))

        # Rewrite total_chunks
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
            c.metadata.total_chunks = total

        return chunks
