"""tests/test_evaluation.py — Basic tests for evaluation utilities."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset_builder import DatasetBuilder


class _FakeLLM:
    def generate(self, prompt: str):
        class _R:
            text = '{"question": "q", "answer": "a"}'
            model = "fake"
        return _R()

    def close(self):
        return None


def test_dataset_builder_dry_run(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        '{"text": "hello", "metadata": {"chunk_id": "c1"}}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "qa.jsonl"
    builder = DatasetBuilder(llm=_FakeLLM())
    pairs = builder.build(str(chunks_path), str(out_path), limit=1, dry_run=True)
    assert pairs
    builder.close()
