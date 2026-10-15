"""Shared helpers for the per-claim custom metrics."""
from __future__ import annotations

import re

YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9_]{2,}|[A-Z]{2,})\b")
_WORD = re.compile(r"[a-z0-9_]{3,}")
_SENT = re.compile(r"(?<=[.!?\n;])\s+")

# Capitalised words that are not entities when they merely start a sentence.
STOPWORDS = frozenset(
    "the this that these those there then when where what which who whom whose why how "
    "and but for with from into onto over under after before during while also however "
    "because since once such some any each every both either neither not none yes "
    "was were are is has have had did does will would could should may might can "
    "its their his her our your they them she him you".split()
)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def content_tokens(text: str) -> set[str]:
    """Lower-cased non-stopword tokens, with years removed."""
    text = YEAR.sub(" ", (text or "").lower())
    return {w for w in _WORD.findall(text) if w not in STOPWORDS}


def aggregate(scores: list[float | None]) -> float | None:
    """Mean over applicable questions; None when no question is applicable."""
    applicable = [s for s in scores if s is not None]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)
