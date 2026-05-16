"""commit_chunker.py — Chunk commit messages (one chunk per commit, max 256 tokens)."""
from __future__ import annotations

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedCommit


class CommitChunker(BaseChunker):
    """One chunk per commit (commit message only).

    The diff content is handled separately by DiffChunker.
    Max 256 tokens per the PRD spec.
    """

    def __init__(self, max_tokens: int = 256) -> None:
        super().__init__(max_tokens=max_tokens)

    def chunk(self, parsed_doc: ParsedCommit, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        commit = parsed_doc

        # Build a concise text representation of the commit message
        text_parts = [commit.message_subject]
        if commit.message_body:
            text_parts.append(commit.message_body)
        if commit.file_paths:
            files_preview = ", ".join(commit.file_paths[:10])
            text_parts.append(f"Files changed: {files_preview}")
        if commit.stats:
            text_parts.append(
                f"Stats: +{commit.stats.get('insertions', 0)} "
                f"-{commit.stats.get('deletions', 0)} "
                f"in {commit.stats.get('files', 0)} file(s)"
            )

        text = "\n\n".join(text_parts)

        # If somehow the message is enormous, truncate at max_chars
        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        metadata = ChunkMetadata(
            doc_type="commit",
            doc_id=commit.sha,
            timestamp=commit.timestamp,
            author=commit.author,
            file_paths=commit.file_paths,
            repo=repo,
            graph_node_id=f"commit_{commit.sha}",
            total_chunks=1,
            chunk_index=0,
        )
        return [Chunk(text=text, metadata=metadata)]
