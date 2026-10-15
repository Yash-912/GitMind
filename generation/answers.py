"""answers.py — Structured answer outcomes, schemas and citation validation.

Callers (direct_qa, decision_memo, api) use these types so an API response can
distinguish a real answer from missing evidence, a provider outage, or model
output that failed validation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_OUTPUT = "invalid_output"


class DirectAnswer(BaseModel):
    answer: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    citations: list[str] = Field(default_factory=list)   # chunk ids
    caveats: list[str] = Field(default_factory=list)


@dataclass
class StructuredResult:
    status: AnswerStatus
    text: str = ""                         # human-readable answer ("" unless answered)
    data: dict[str, Any] | None = None     # validated structured payload
    model: str = ""
    provider: str = ""
    retrieved_ids: list[str] = field(default_factory=list)   # candidates found
    included_ids: list[str] = field(default_factory=list)    # shown to the model
    cited_ids: list[str] = field(default_factory=list)       # validated citations
    invalid_citations: list[str] = field(default_factory=list)
    error: str = ""
    attempts: int = 0


INSUFFICIENT_MESSAGE = (
    "No retrieved evidence was available in the selected scope, so no answer was generated."
)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw model output, tolerating fenced/prefixed text."""
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def split_citations(cited: list[str], included_ids: list[str]) -> tuple[list[str], list[str]]:
    """Return (valid, invalid). A citation is valid only if the model was shown it."""
    allowed = set(included_ids)
    valid, invalid = [], []
    for c in dict.fromkeys(x.strip() for x in cited if x and x.strip()):
        (valid if c in allowed else invalid).append(c)
    return valid, invalid


def validate_direct(raw: str, included_ids: list[str]) -> tuple[DirectAnswer | None, list[str], str]:
    """Parse + validate a direct answer. Returns (answer, invalid_citations, error)."""
    obj = extract_json_object(raw)
    if obj is None:
        return None, [], "output was not a JSON object"
    try:
        parsed = DirectAnswer.model_validate(obj)
    except ValidationError as exc:
        err = exc.errors()[0]
        return None, [], f"schema violation: {err['loc']} {err['msg']}"
    valid, invalid = split_citations(parsed.citations, included_ids)
    if invalid:
        return None, invalid, f"cited unknown chunk ids: {invalid}"
    if not valid:
        return None, [], "answer cited no evidence"
    parsed.citations = valid
    return parsed, [], ""
