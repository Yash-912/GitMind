from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from pydantic import BaseModel, ValidationError

from .answers import (
    INSUFFICIENT_MESSAGE,
    AnswerStatus,
    StructuredResult,
    extract_json_object,
    split_citations,
)
from .llm_client import LLMClient, LLMUnavailableError


class DecisionMemo(BaseModel):
    decision: str
    date: str | None = None
    decision_makers: list[str] = []
    context: str
    alternatives: list[str] = []
    rationale: str
    consequences: list[str] = []
    evidence: list[str] = []


@dataclass
class DecisionMemoResult:
    raw_text: str
    model: str
    memo: DecisionMemo | None = None


class DecisionMemoGenerator:
    def __init__(
        self, llm: LLMClient | None = None, strict_json: bool = True, max_retries: int = 1
    ) -> None:
        self.llm = llm or LLMClient()
        self.strict_json = strict_json
        self.max_retries = max_retries
        template_path = Path(__file__).parent / "prompt_templates" / "decision_memo.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(self, question: str, context: str) -> DecisionMemoResult:
        """Legacy path. Prefer generate_structured."""
        prompt = self._template.render(question=question, context=context)
        resp = self.llm.generate(prompt)
        memo = self._parse_json(resp.text)
        if memo is None and self.strict_json:
            raise ValueError("Decision memo output was not valid JSON")
        return DecisionMemoResult(raw_text=resp.text, model=resp.model, memo=memo)

    def generate_structured(
        self,
        question: str,
        context: str,
        included_ids: list[str],
        retrieved_ids: list[str] | None = None,
    ) -> StructuredResult:
        """Validated memo whose evidence ids were all shown to the model."""
        retrieved = list(retrieved_ids if retrieved_ids is not None else included_ids)
        base = dict(retrieved_ids=retrieved, included_ids=list(included_ids))
        if not included_ids or not context.strip():
            return StructuredResult(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE, text=INSUFFICIENT_MESSAGE, **base
            )

        previous_error, last_error, last_invalid = "", "", []
        model = provider = ""
        for attempt in range(1, self.max_retries + 2):
            prompt = self._template.render(
                question=question, context=context, previous_error=previous_error
            )
            try:
                resp = self.llm.generate(prompt)
            except LLMUnavailableError as exc:
                return StructuredResult(
                    status=AnswerStatus.PROVIDER_UNAVAILABLE, error=str(exc),
                    model=model, provider=provider, attempts=attempt, **base,
                )
            model, provider = getattr(resp, "model", ""), getattr(resp, "provider", "")
            memo, invalid, error = self._validate(resp.text, included_ids)
            if memo is not None:
                return StructuredResult(
                    status=AnswerStatus.ANSWERED,
                    text=memo.decision,
                    data=memo.model_dump(),
                    model=model, provider=provider,
                    cited_ids=memo.evidence, attempts=attempt, **base,
                )
            last_error, last_invalid, previous_error = error, invalid, error

        return StructuredResult(
            status=AnswerStatus.INVALID_OUTPUT, error=last_error,
            invalid_citations=last_invalid, model=model, provider=provider,
            attempts=self.max_retries + 1, **base,
        )

    @staticmethod
    def _validate(raw: str, included_ids: list[str]):
        obj = extract_json_object(raw)
        if obj is None:
            return None, [], "output was not a JSON object"
        try:
            memo = DecisionMemo.model_validate(obj)
        except ValidationError as exc:
            err = exc.errors()[0]
            return None, [], f"schema violation: {err['loc']} {err['msg']}"
        valid, invalid = split_citations(memo.evidence, included_ids)
        if invalid:
            return None, invalid, f"evidence cites unknown chunk ids: {invalid}"
        if not valid:
            return None, [], "memo cited no evidence"
        memo.evidence = valid
        return memo, [], ""

    def _parse_json(self, text: str) -> DecisionMemo | None:
        try:
            data = json.loads(text)
            return DecisionMemo.model_validate(data)
        except Exception:
            pass

        # Try to extract first JSON object from text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return DecisionMemo.model_validate(data)
            except Exception:
                return None
        return None

    def close(self) -> None:
        self.llm.close()
