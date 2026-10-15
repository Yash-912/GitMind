from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .types import CandidateChunk

_SEPARATOR = "\n---\n"
_TRUNC_MARK = "\n[truncated]\n"


@dataclass
class AssembledContext:
    """Rendered context plus an exact record of what the model will see."""

    text: str
    included_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)      # did not fit the budget
    truncated_ids: list[str] = field(default_factory=list)    # included in shortened form
    tokens: int = 0


class ContextAssembler:
    """Assemble a context window within a token budget.

    The budget covers the complete rendered string, separators included. A
    chunk that does not fit is skipped (later, smaller chunks can still be
    used). A chunk too large for the whole budget is shortened rather than
    silently emptying the context.
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        chars_per_token: int = 4,
        order: str = "rank",
    ) -> None:
        if order not in {"rank", "chronological"}:
            raise ValueError("order must be 'rank' or 'chronological'")
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.order = order
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

    @staticmethod
    def _header(c: CandidateChunk) -> str:
        m = c.metadata
        return (
            f"[{m.get('doc_type', '')}] "
            f"[{m.get('doc_id', '')}] "
            f"[{m.get('timestamp', '')}] "
            f"[{m.get('author', '')}] "
            f"[chunk:{c.chunk_id}]\n"
        )

    def _shorten(self, header: str, text: str, token_budget: int) -> str:
        """Cut `text` so header + text + marker fits `token_budget` tokens."""
        lo, hi = 0, len(text)
        while lo < hi:  # longest prefix whose full rendering fits
            mid = (lo + hi + 1) // 2
            rendered = header + text[:mid].rstrip() + _TRUNC_MARK
            if self._count_tokens(rendered) <= token_budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + _TRUNC_MARK if lo else ""

    def assemble_with_report(self, chunks: Iterable[CandidateChunk]) -> AssembledContext:
        sep_tokens = self._count_tokens(_SEPARATOR)
        blocks: list[tuple[CandidateChunk, str]] = []
        skipped: list[str] = []
        truncated: list[str] = []
        used = 0

        for c in chunks:
            header = self._header(c)
            block = header + c.text + "\n"
            cost = self._count_tokens(block) + (sep_tokens if blocks else 0)
            if used + cost <= self.max_tokens:
                blocks.append((c, block))
                used += cost
                continue

            remaining = self.max_tokens - used - (sep_tokens if blocks else 0)
            # Only shorten when a chunk could never fit whole; otherwise skip it.
            never_fits = self._count_tokens(block) > self.max_tokens
            if never_fits and remaining >= self.max_tokens // 4:
                # Cap an oversized chunk at half the budget so it cannot starve the rest.
                body = self._shorten(header, c.text, min(remaining, self.max_tokens // 2))
                if body:
                    short = header + body
                    blocks.append((c, short))
                    truncated.append(c.chunk_id)
                    used += self._count_tokens(short) + (sep_tokens if len(blocks) > 1 else 0)
                    continue
            skipped.append(c.chunk_id)

        if self.order == "chronological":
            blocks.sort(key=lambda cb: str(cb[0].metadata.get("timestamp", "")))

        text = _SEPARATOR.join(b for _, b in blocks)
        return AssembledContext(
            text=text,
            included_ids=[c.chunk_id for c, _ in blocks],
            skipped_ids=skipped,
            truncated_ids=truncated,
            tokens=self._count_tokens(text) if text else 0,
        )

    def assemble(self, chunks: Iterable[CandidateChunk]) -> str:
        return self.assemble_with_report(chunks).text
