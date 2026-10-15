from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jinja2 import Template

from .llm_client import LLMClient
from .statistics import RiskStats, compute_risk, render_stats


@dataclass
class RiskReportResult:
    raw_text: str
    model: str
    stats: RiskStats | None = None


class RiskReportGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        template_path = Path(__file__).parent / "prompt_templates" / "risk_report.j2"
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def generate(
        self,
        module_name: str,
        context: str,
        *,
        records: Iterable[dict] | None = None,
        scope=None,
        coverage_complete: bool = False,
    ) -> RiskReportResult:
        """If `records` (all chunk records in scope) are given, risk figures are
        computed from them and the LLM only narrates."""
        stats = None
        if records is not None:
            stats = compute_risk(
                records, module_name, scope, coverage_complete=coverage_complete
            )
        prompt = self._template.render(
            module_name=module_name,
            context=context,
            stats=render_stats(stats) if stats else "",
        )
        resp = self.llm.generate(prompt)
        return RiskReportResult(raw_text=resp.text, model=resp.model, stats=stats)

    def close(self) -> None:
        self.llm.close()
