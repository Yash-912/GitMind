from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from .llm_client import LLMClient


@dataclass
class BlameMapResult:
    raw_text: str
    model: str


class BlameMapGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        template_path = Path(__file__).parent / "prompt_templates" / "blame_map.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(self, module_name: str, context: str) -> BlameMapResult:
        prompt = self._template.render(module_name=module_name, context=context)
        resp = self.llm.generate(prompt)
        return BlameMapResult(raw_text=resp.text, model=resp.model)

    def close(self) -> None:
        self.llm.close()
