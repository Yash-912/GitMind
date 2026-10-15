"""Phase 4 tests: provider outcomes, context accounting, structured answers."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from generation.answers import (
    AnswerStatus,
    extract_json_object,
    split_citations,
    validate_direct,
)
from generation.direct_qa import DirectQAGenerator
from generation.llm_client import LLMClient, LLMResponse, LLMUnavailableError
from retrieval.context_assembler import ContextAssembler
from retrieval.types import CandidateChunk


# ------------------------------ LLM client ------------------------------ #

class _Gemini:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents):
        self.calls += 1
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(text=b)


def _client(gemini=None, ollama=None, **kw):
    c = LLMClient(**kw)
    c._gemini = gemini
    c._sleep = lambda s: None
    if ollama is not None:
        c._ollama_generate = ollama
    return c


def _ollama_down(prompt):
    raise ConnectionError("refused")


def test_total_outage_raises_not_empty_success():
    client = _client(gemini=_Gemini([RuntimeError("boom")]), ollama=_ollama_down)
    with pytest.raises(LLMUnavailableError) as exc:
        client.generate("hi")
    assert any("gemini" in c for c in exc.value.causes)
    assert any("ollama" in c for c in exc.value.causes)


def test_empty_responses_are_failures():
    client = _client(gemini=_Gemini([""]), ollama=lambda p: "   ")
    with pytest.raises(LLMUnavailableError):
        client.generate("hi")


def test_primary_success_reports_provider():
    r = _client(gemini=_Gemini(["answer"])).generate("hi")
    assert (r.text, r.provider, r.fallback_used) == ("answer", "gemini", False)


def test_fallback_to_ollama_is_reported():
    client = _client(gemini=_Gemini([RuntimeError("down")]), ollama=lambda p: "local")
    r = client.generate("hi")
    assert (r.text, r.provider, r.fallback_used) == ("local", "ollama", True)


def test_rate_limit_retries_then_succeeds():
    g = _Gemini([RuntimeError("429 RESOURCE_EXHAUSTED"), "ok"])
    r = _client(gemini=g).generate("hi")
    assert r.text == "ok" and g.calls == 2


def test_deadline_stops_retry_backoff():
    g = _Gemini([RuntimeError("429")] * 5)
    client = _client(gemini=g, ollama=_ollama_down, deadline_s=3)
    with pytest.raises(LLMUnavailableError) as exc:
        client.generate("hi")
    assert g.calls == 1  # first back-off (5s) exceeds the 3s budget
    assert any("rate limited" in c for c in exc.value.causes)


def test_ollama_only_when_no_gemini_key():
    r = _client(gemini=None, ollama=lambda p: "local").generate("hi")
    assert r.provider == "ollama" and r.fallback_used is False


# --------------------------- context assembler --------------------------- #

def _cand(i, text, ts="2022-01-01"):
    return CandidateChunk(
        chunk_id=f"c{i}", score=1.0, source="t", text=text,
        metadata={"doc_type": "pr", "doc_id": str(i), "timestamp": ts, "author": "a"},
    )


def _asm(**kw):
    asm = ContextAssembler(**kw)
    asm._tokenizer = None  # deterministic len//4 counting
    return asm


def test_oversized_first_chunk_does_not_empty_context():
    asm = _asm(max_tokens=100)
    out = asm.assemble_with_report([_cand(1, "x" * 4000), _cand(2, "small useful text")])
    assert "c2" in out.included_ids
    assert "c1" in out.skipped_ids or "c1" in out.truncated_ids
    assert out.text


def test_never_fitting_chunk_is_truncated_when_alone():
    asm = _asm(max_tokens=100)
    out = asm.assemble_with_report([_cand(1, "word " * 1000)])
    assert out.included_ids == ["c1"] and out.truncated_ids == ["c1"]
    assert asm._count_tokens(out.text) <= 100
    assert "[truncated]" in out.text


def test_budget_counts_separators_and_headers():
    asm = _asm(max_tokens=60)
    out = asm.assemble_with_report([_cand(i, "y" * 60) for i in range(10)])
    assert asm._count_tokens(out.text) <= 60
    assert len(out.included_ids) + len(out.skipped_ids) == 10


def test_included_ids_match_rendered_text():
    asm = _asm(max_tokens=200)
    out = asm.assemble_with_report([_cand(i, f"text {i}") for i in range(3)])
    for cid in out.included_ids:
        assert f"[chunk:{cid}]" in out.text
    assert asm.assemble([_cand(1, "t")]) != ""


def test_chronological_order_option():
    asm = _asm(max_tokens=500, order="chronological")
    out = asm.assemble_with_report([_cand(1, "late", "2023-01-01"), _cand(2, "early", "2020-01-01")])
    assert out.text.index("early") < out.text.index("late")


def test_invalid_order_rejected():
    with pytest.raises(ValueError):
        ContextAssembler(order="random")


# ------------------------------ structured QA ------------------------------ #

class _LLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, model="fake", provider="fake")

    def close(self):
        pass


def _answer(**kw):
    base = {"answer": "It was added in PR 7.", "confidence": "high",
            "citations": ["c1"], "caveats": []}
    base.update(kw)
    return json.dumps(base)


def test_no_evidence_skips_llm():
    llm = _LLM()
    r = DirectQAGenerator(llm).generate_structured("q", "", [])
    assert r.status is AnswerStatus.INSUFFICIENT_EVIDENCE and llm.prompts == []


def test_valid_answer_with_included_citation():
    r = DirectQAGenerator(_LLM(_answer())).generate_structured(
        "q", "ctx", ["c1", "c2"], retrieved_ids=["c1", "c2", "c3"])
    assert r.status is AnswerStatus.ANSWERED
    assert r.cited_ids == ["c1"] and r.retrieved_ids == ["c1", "c2", "c3"]
    assert r.included_ids == ["c1", "c2"] and r.data["confidence"] == "high"


def test_forged_citation_rejected_then_retried():
    llm = _LLM(_answer(citations=["ghost"]), _answer(citations=["c1"]))
    r = DirectQAGenerator(llm).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.ANSWERED and r.attempts == 2
    assert "ghost" in llm.prompts[1]  # error fed back to the model


def test_forged_citation_persisting_is_invalid_output():
    llm = _LLM(_answer(citations=["ghost"]), _answer(citations=["ghost"]))
    r = DirectQAGenerator(llm).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.INVALID_OUTPUT and r.invalid_citations == ["ghost"]
    assert r.text == ""


def test_malformed_output_is_invalid_not_answer():
    r = DirectQAGenerator(_LLM("not json", "still not")).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.INVALID_OUTPUT


def test_answer_without_citations_rejected():
    llm = _LLM(_answer(citations=[]), _answer(citations=[]))
    r = DirectQAGenerator(llm).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.INVALID_OUTPUT


def test_bad_confidence_rejected():
    parsed, _, err = validate_direct(_answer(confidence="certain"), ["c1"])
    assert parsed is None and "schema" in err


def test_provider_outage_status():
    llm = _LLM(LLMUnavailableError(["gemini: x", "ollama: y"]))
    r = DirectQAGenerator(llm).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.PROVIDER_UNAVAILABLE and "ollama" in r.error


def test_json_extraction_variants():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert extract_json_object("[1,2]") is None


def test_split_citations_dedupes():
    assert split_citations(["a", "a", "z", " "], ["a"]) == (["a"], ["z"])
