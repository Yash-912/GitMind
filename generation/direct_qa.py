from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from .llm_client import LLMClient


@dataclass
class DirectQAResult:
    answer: str
    model: str
    evidence: list[str]


class DirectQAGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        template_path = Path(__file__).parent / "prompt_templates" / "direct_qa.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(self, question: str, context: str) -> DirectQAResult:
        prompt = self._template.render(question=question, context=context)
        resp = self.llm.generate(prompt)
        evidence = self._extract_evidence(context)
        return DirectQAResult(answer=resp.text, model=resp.model, evidence=evidence)

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
