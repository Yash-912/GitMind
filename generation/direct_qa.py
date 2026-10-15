from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from .answers import (
    INSUFFICIENT_MESSAGE,
    AnswerStatus,
    StructuredResult,
    validate_direct,
)
from .llm_client import LLMClient, LLMUnavailableError


@dataclass
class DirectQAResult:
    answer: str
    model: str
    evidence: list[str]


class DirectQAGenerator:
    def __init__(self, llm: LLMClient | None = None, max_retries: int = 1) -> None:
        self.llm = llm or LLMClient()
        self.max_retries = max_retries
        base = Path(__file__).parent / "prompt_templates"
        self._template = Template((base / "direct_qa.j2").read_text(encoding="utf-8"))
        self._json_template = Template((base / "direct_qa_json.j2").read_text(encoding="utf-8"))

    def generate(self, question: str, context: str) -> DirectQAResult:
        """Legacy free-text answer. Prefer generate_structured."""
        prompt = self._template.render(question=question, context=context)
        resp = self.llm.generate(prompt)
        evidence = self._extract_evidence(context)
        return DirectQAResult(answer=resp.text, model=resp.model, evidence=evidence)

    def generate_structured(
        self,
        question: str,
        context: str,
        included_ids: list[str],
        retrieved_ids: list[str] | None = None,
    ) -> StructuredResult:
        """Validated answer with checked citations and an explicit outcome status.

        - No included evidence: deterministic insufficient_evidence, no LLM call.
        - Provider outage: provider_unavailable.
        - Malformed JSON, schema violation or citations the model was never
          shown: retried with the error fed back, then invalid_output.
        """
        retrieved = list(retrieved_ids if retrieved_ids is not None else included_ids)
        base = dict(retrieved_ids=retrieved, included_ids=list(included_ids))
        if not included_ids or not context.strip():
            return StructuredResult(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE, text=INSUFFICIENT_MESSAGE, **base
            )

        previous_error = ""
        last_error, last_invalid = "", []
        model = provider = ""
        for attempt in range(1, self.max_retries + 2):
            prompt = self._json_template.render(
                question=question, context=context, previous_error=previous_error
            )
            try:
                resp = self.llm.generate(prompt)
            except LLMUnavailableError as exc:
                return StructuredResult(
                    status=AnswerStatus.PROVIDER_UNAVAILABLE,
                    error=str(exc),
                    model=model,
                    provider=provider,
                    attempts=attempt,
                    **base,
                )
            model = getattr(resp, "model", "")
            provider = getattr(resp, "provider", "")
            parsed, invalid, error = validate_direct(resp.text, included_ids)
            if parsed is not None:
                return StructuredResult(
                    status=AnswerStatus.ANSWERED,
                    text=parsed.answer,
                    data=parsed.model_dump(),
                    model=model,
                    provider=provider,
                    cited_ids=parsed.citations,
                    attempts=attempt,
                    **base,
                )
            last_error, last_invalid = error, invalid
            previous_error = error

        return StructuredResult(
            status=AnswerStatus.INVALID_OUTPUT,
            error=last_error,
            invalid_citations=last_invalid,
            model=model,
            provider=provider,
            attempts=self.max_retries + 1,
            **base,
        )

    def _extract_evidence(self, context: str) -> list[str]:
        results: list[str] = []
        for line in context.splitlines():
            if line.startswith("[") and "]" in line:
                parts = re.findall(r"\[([^\]]+)\]", line)
                if len(parts) >= 2:
                    results.append(f"{parts[0]}:{parts[1]}")
        return list(dict.fromkeys(results))

    def close(self) -> None:
        self.llm.close()
