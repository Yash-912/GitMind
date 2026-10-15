"""log_chunker.py — Chunk CI/CD logs or GraphQL PR records."""
from __future__ import annotations

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata


class LogChunker(BaseChunker):
    """Generic chunker for log-like documents."""

    def __init__(
        self, doc_type: str, max_tokens: int = 600, node_type: str | None = None
    ) -> None:
        super().__init__(max_tokens=max_tokens)
        self.doc_type = doc_type
        # Graph node namespace; a GraphQL PR record is node ``pr_<n>``, never
        # ``pr_graphql_<n>``.
        self.node_type = node_type or ("pr" if doc_type == "pr_graphql" else doc_type)

    def chunk(self, parsed_doc, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        title = getattr(parsed_doc, "title", "") or getattr(parsed_doc, "name", "")
        body = getattr(parsed_doc, "body_clean", "")
        doc_id = str(getattr(parsed_doc, "run_id", "") or getattr(parsed_doc, "number", ""))
        author = getattr(parsed_doc, "actor", "") or getattr(parsed_doc, "author", "")
        timestamp = getattr(parsed_doc, "created_at", None)

        header = f"[{self.doc_type}] {title}".strip()
        text = f"{header}\n\n{body}".strip()
        closing = getattr(parsed_doc, "closing_issue_numbers", None) or []
        if closing:
            incomplete = getattr(parsed_doc, "closing_issues_complete", True) is False
            text += (
                "\n\nCloses: " + ", ".join(f"#{n}" for n in closing)
                + (" (list may be incomplete)" if incomplete else "")
            )

        chunks: list[Chunk] = []
        for s, e in self._split_spans(text) or [(0, 0)]:
            chunks.append(
                Chunk(
                    text=text[s:e],
                    metadata=ChunkMetadata(
                        doc_type=self.doc_type,
                        doc_id=doc_id,
                        timestamp=timestamp or __import__("datetime").datetime.utcnow(),
                        author=author,
                        repo=repo,
                        graph_node_id=f"{self.node_type}_{doc_id}",
                        char_start=s,
                        char_end=e,
                    ),
                )
            )

        total = len(chunks)
        for idx, c in enumerate(chunks):
            c.metadata.chunk_index = idx
            c.metadata.total_chunks = total
        return chunks
