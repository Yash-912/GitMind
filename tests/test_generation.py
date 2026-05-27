"""tests/test_generation.py — Basic tests for generation layer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.direct_qa import DirectQAGenerator
from generation.decision_memo import DecisionMemoGenerator
from generation.blame_map import BlameMapGenerator
from generation.risk_report import RiskReportGenerator


class _FakeLLM:
    def generate(self, prompt: str):
        class _R:
            text = "ok"
            model = "fake"
        return _R()

    def close(self):
        return None


def test_direct_qa_template_loads():
    gen = DirectQAGenerator(llm=_FakeLLM())
    result = gen.generate("q", "ctx")
    assert result.answer
    gen.close()


def test_decision_memo_template_loads():
    gen = DecisionMemoGenerator(llm=_FakeLLM(), strict_json=False)
    result = gen.generate("q", "ctx")
    assert result.raw_text
    gen.close()


def test_blame_map_template_loads():
    gen = BlameMapGenerator(llm=_FakeLLM())
    result = gen.generate("module", "ctx")
    assert result.raw_text
    gen.close()


def test_risk_report_template_loads():
    gen = RiskReportGenerator(llm=_FakeLLM())
    result = gen.generate("module", "ctx")
    assert result.raw_text
    gen.close()
