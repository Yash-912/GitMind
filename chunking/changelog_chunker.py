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
    # Offsets into the text passed to parse_changelog()
    start: int = 0          # start of the heading
    end: int = 0            # end of the section (start of next heading / EOF)
    body_start: int = 0     # where ``body`` (stripped) begins


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
    """Split a full CHANGELOG text into version sections (with source offsets)."""
    matches = list(_VERSION_HEADING.finditer(text))
    sections: list[ChangelogSection] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[m.end():end]
        sections.append(
            ChangelogSection(
                heading=heading,
                body=raw.strip(),
                version=_parse_version_from_heading(heading),
                date=_parse_date_from_heading(heading),
                start=m.start(),
                end=end,
                body_start=m.end() + (len(raw) - len(raw.lstrip())),
            )
        )
    return sections


class ChangelogChunker(BaseChunker):
    """One chunk per version section in a CHANGELOG or release notes body.

    Sections longer than max_tokens are split losslessly with overlap; every
    piece keeps its own deterministic ID and char_start/char_end offsets into
    the release body.
    """

    def __init__(self, max_tokens: int = 600) -> None:
        super().__init__(max_tokens=max_tokens)

    def chunk(self, parsed_doc: ParsedRelease, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        release = parsed_doc
        body = release.body_clean or ""

        # Try to split into sub-sections (useful for aggregated CHANGELOG files)
        sections = parse_changelog(body)
        if not sections:
            # Single release note: split only if over the limit
            header = f"[Release {release.tag}]\n\n"
            chunks: list[Chunk] = []
            for text, s, e in self._split_with_header(header, body):
                meta = ChunkMetadata(
                    doc_type="release",
                    doc_id=release.tag,
                    timestamp=release.timestamp or datetime.utcnow(),
                    repo=repo,
                    graph_node_id=f"release_{release.tag}",
                    char_start=s,
                    char_end=e,
                )
                chunks.append(Chunk(text=text, metadata=meta))
            return self._reindex(chunks)

        chunks = []
        # Text before the first version heading is kept as a preamble chunk
        # (a bare "# Changelog" title carries no information and is skipped).
        preamble = body[: sections[0].start].strip()
        if any(ln.strip() and not ln.lstrip().startswith("#") for ln in preamble.splitlines()):
            header = f"[Release {release.tag} preamble]\n\n"
            offset = body.index(preamble)
            for text, s, e in self._split_with_header(header, preamble):
                meta = ChunkMetadata(
                    doc_type="release",
                    doc_id=release.tag,
                    timestamp=release.timestamp or datetime.utcnow(),
                    repo=repo,
                    graph_node_id=f"release_{release.tag}",
                    char_start=offset + s,
                    char_end=offset + e,
                )
                chunks.append(Chunk(text=text, metadata=meta))
        for section in sections:
            header = f"[Release {section.version}]\n{section.heading}\n\n"
            ts = section.date or release.timestamp or datetime.utcnow()
            for text, s, e in self._split_with_header(header, section.body):
                meta = ChunkMetadata(
                    doc_type="release",
                    doc_id=section.version,
                    timestamp=ts,
                    repo=repo,
                    graph_node_id=f"release_{section.version}",
                    char_start=section.body_start + s,
                    char_end=section.body_start + e,
                )
                chunks.append(Chunk(text=text, metadata=meta))
        return self._reindex(chunks)

    @staticmethod
    def _reindex(chunks: list[Chunk]) -> list[Chunk]:
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
            c.metadata.total_chunks = total
        return chunks
