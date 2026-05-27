from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from pydantic import BaseModel

from .llm_client import LLMClient


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
    def __init__(self, llm: LLMClient | None = None, strict_json: bool = True) -> None:
        self.llm = llm or LLMClient()
        self.strict_json = strict_json
        template_path = Path(__file__).parent / "prompt_templates" / "decision_memo.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(self, question: str, context: str) -> DecisionMemoResult:
        prompt = self._template.render(question=question, context=context)
        resp = self.llm.generate(prompt)
        memo = self._parse_json(resp.text)
        if memo is None and self.strict_json:
            raise ValueError("Decision memo output was not valid JSON")
        return DecisionMemoResult(raw_text=resp.text, model=resp.model, memo=memo)

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
