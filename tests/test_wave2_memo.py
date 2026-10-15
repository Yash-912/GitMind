"""Decision memo: schema validation and evidence-id checking."""
import json

from generation.answers import AnswerStatus
from generation.decision_memo import DecisionMemoGenerator
from generation.llm_client import LLMResponse, LLMUnavailableError


class _LLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, model="fake", provider="fake")

    def close(self):
        pass


def _memo(**kw):
    base = {
        "decision": "Adopt JWT", "date": "2022-03-01", "decision_makers": ["alice"],
        "context": "Sessions did not scale", "alternatives": ["sessions"],
        "rationale": "Stateless", "consequences": ["token revocation"],
        "evidence": ["c1"],
    }
    base.update(kw)
    return json.dumps(base)


def test_valid_memo_with_shown_evidence():
    r = DecisionMemoGenerator(_LLM(_memo())).generate_structured("why jwt", "ctx", ["c1", "c2"])
    assert r.status is AnswerStatus.ANSWERED and r.cited_ids == ["c1"]
    assert r.data["decision"] == "Adopt JWT" and r.text == "Adopt JWT"


def test_unknown_evidence_rejected_then_retried():
    llm = _LLM(_memo(evidence=["ghost"]), _memo(evidence=["c2"]))
    r = DecisionMemoGenerator(llm).generate_structured("q", "ctx", ["c1", "c2"])
    assert r.status is AnswerStatus.ANSWERED and r.attempts == 2
    assert "ghost" in llm.prompts[1]


def test_persistent_forged_evidence_is_invalid_output():
    llm = _LLM(_memo(evidence=["ghost"]), _memo(evidence=["ghost"]))
    r = DecisionMemoGenerator(llm).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.INVALID_OUTPUT and r.invalid_citations == ["ghost"]


def test_memo_without_evidence_rejected():
    llm = _LLM(_memo(evidence=[]), _memo(evidence=[]))
    assert DecisionMemoGenerator(llm).generate_structured("q", "ctx", ["c1"]).status is AnswerStatus.INVALID_OUTPUT


def test_missing_required_field_rejected():
    bad = json.dumps({"decision": "x", "evidence": ["c1"]})
    r = DecisionMemoGenerator(_LLM(bad, bad)).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.INVALID_OUTPUT and "schema" in r.error


def test_no_evidence_does_not_call_llm():
    llm = _LLM()
    r = DecisionMemoGenerator(llm).generate_structured("q", "", [])
    assert r.status is AnswerStatus.INSUFFICIENT_EVIDENCE and llm.prompts == []


def test_provider_outage():
    r = DecisionMemoGenerator(_LLM(LLMUnavailableError(["x"]))).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.PROVIDER_UNAVAILABLE


def test_fenced_json_accepted():
    r = DecisionMemoGenerator(_LLM("```json\n" + _memo() + "\n```")).generate_structured("q", "ctx", ["c1"])
    assert r.status is AnswerStatus.ANSWERED


def test_legacy_generate_still_works():
    g = DecisionMemoGenerator(_LLM(_memo()), strict_json=True)
    assert g.generate("q", "ctx").memo.decision == "Adopt JWT"
