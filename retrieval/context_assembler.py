from __future__ import annotations

from typing import Iterable

from .types import CandidateChunk


class ContextAssembler:
    """Assemble a context window within a token budget."""

    def __init__(self, max_tokens: int = 3000, chars_per_token: int = 4) -> None:
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            import tiktoken  # type: ignore

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // self.chars_per_token)

    def assemble(self, chunks: Iterable[CandidateChunk]) -> str:
        parts: list[str] = []
        used = 0
        for c in chunks:
            header = (
                f"[{c.metadata.get('doc_type', '')}] "
                f"[{c.metadata.get('doc_id', '')}] "
                f"[{c.metadata.get('timestamp', '')}] "
                f"[{c.metadata.get('author', '')}]\n"
            )
            block = header + c.text + "\n"
            tokens = self._count_tokens(block)
            if used + tokens > self.max_tokens:
                break
            parts.append(block)
            used += tokens
        return "\n---\n".join(parts)
