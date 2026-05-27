from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from generation.llm_client import LLMClient


@dataclass
class QAPair:
    question: str
    answer: str
    context: str
    source_chunk_id: str
    qa_type: str = "standard"


class DatasetBuilder:
    """Build a lightweight QA dataset from existing chunks."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def build(
        self,
        chunks_path: str,
        out_path: str,
        limit: int = 100,
        dry_run: bool = False,
        include_adversarial: bool = True,
        include_temporal: bool = True,
        include_multihop: bool = True,
    ) -> list[QAPair]:
        chunks = self._load_chunks(chunks_path)
        if not chunks:
            return []

        sampled = random.sample(chunks, min(limit, len(chunks)))
        pairs: list[QAPair] = []
        for record in sampled:
            text = record.get("text", "")
            chunk_id = record.get("metadata", {}).get("chunk_id", "")
            if not text:
                continue

            if dry_run:
                question = "What does this change describe?"
                answer = text[:200]
            else:
                prompt = (
                    "Create one concise question and answer based on the context below.\n\n"
                    f"Context:\n{text}\n\n"
                    "Return as JSON with fields: question, answer."
                )
                resp = self.llm.generate(prompt).text
                try:
                    data = json.loads(resp)
                    question = data.get("question", "")
                    answer = data.get("answer", "")
                except Exception:
                    question = "What happened in this context?"
                    answer = text[:200]

            pairs.append(QAPair(question=question, answer=answer, context=text, source_chunk_id=chunk_id, qa_type="standard"))

            if include_temporal:
                q, a = self._temporal_pair(text, dry_run=dry_run)
                pairs.append(QAPair(question=q, answer=a, context=text, source_chunk_id=chunk_id, qa_type="temporal"))

            if include_multihop:
                q, a = self._multihop_pair(text, dry_run=dry_run)
                pairs.append(QAPair(question=q, answer=a, context=text, source_chunk_id=chunk_id, qa_type="multihop"))

        if include_adversarial:
            pairs.extend(self._adversarial_pairs(sampled, dry_run=dry_run))

        self._write_pairs(out_path, pairs)
        return pairs

    def _load_chunks(self, chunks_path: str) -> list[dict]:
        path = Path(chunks_path)
        if not path.exists():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _write_pairs(self, out_path: str, pairs: list[QAPair]) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")

    def _temporal_pair(self, text: str, dry_run: bool = False) -> tuple[str, str]:
        if dry_run:
            return "When did this happen?", "Unknown"
        prompt = (
            "Create a temporal question and answer based on the context below. "
            "Return JSON with fields: question, answer.\n\n"
            f"Context:\n{text}\n"
        )
        return self._qa_from_prompt(prompt, fallback_q="When did this happen?")

    def _multihop_pair(self, text: str, dry_run: bool = False) -> tuple[str, str]:
        if dry_run:
            return "What led to this change?", "Unknown"
        prompt = (
            "Create a multi-hop question that requires linking context details. "
            "Return JSON with fields: question, answer.\n\n"
            f"Context:\n{text}\n"
        )
        return self._qa_from_prompt(prompt, fallback_q="What led to this change?")

    def _adversarial_pairs(self, sampled: list[dict], dry_run: bool = False) -> list[QAPair]:
        pairs: list[QAPair] = []
        for record in sampled[: max(1, len(sampled) // 10)]:
            text = record.get("text", "")
            chunk_id = record.get("metadata", {}).get("chunk_id", "")
            if dry_run:
                q = "Is there evidence of a database migration in this context?"
                a = "I could not find evidence in the provided context."
            else:
                prompt = (
                    "Create a question that is NOT answered by the context below. "
                    "Return JSON with fields: question, answer (answer should say no evidence).\n\n"
                    f"Context:\n{text}\n"
                )
                q, a = self._qa_from_prompt(prompt, fallback_q="Is there evidence not in context?")
            pairs.append(QAPair(question=q, answer=a, context=text, source_chunk_id=chunk_id, qa_type="adversarial"))
        return pairs

    def _qa_from_prompt(self, prompt: str, fallback_q: str) -> tuple[str, str]:
        resp = self.llm.generate(prompt).text
        try:
            data = json.loads(resp)
            return data.get("question", fallback_q), data.get("answer", "")
        except Exception:
            return fallback_q, ""

    def close(self) -> None:
        self.llm.close()
