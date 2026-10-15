from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jinja2 import Template

from .llm_client import LLMClient
from .statistics import OwnershipStats, compute_ownership, render_stats


@dataclass
class BlameMapResult:
    raw_text: str
    model: str
    stats: OwnershipStats | None = None


class BlameMapGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        template_path = Path(__file__).parent / "prompt_templates" / "blame_map.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(
        self,
        module_name: str,
        context: str,
        *,
        records: Iterable[dict] | None = None,
        scope=None,
        coverage_complete: bool = False,
    ) -> BlameMapResult:
        """If `records` (all chunk records in scope) are given, ownership figures
        are computed from them and the LLM only narrates."""
        stats = None
        if records is not None:
            stats = compute_ownership(
                records, module_name, scope, coverage_complete=coverage_complete
            )
        prompt = self._template.render(
            module_name=module_name,
            context=context,
            stats=render_stats(stats) if stats else "",
        )
        resp = self.llm.generate(prompt)
        return BlameMapResult(raw_text=resp.text, model=resp.model, stats=stats)

    def close(self) -> None:
        self.llm.close()
