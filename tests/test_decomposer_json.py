"""Query decomposer must accept JSON wrapped in a markdown fence (Gemini's usual reply shape)."""
from types import SimpleNamespace

from retrieval.query_decomposer import QueryDecomposer


class _LLM:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return SimpleNamespace(text=self.text, model="fake")

    def close(self):
        pass


def test_fenced_json_plan_is_used_without_warning():
    reply = '```json\n{"entities": ["JWT"], "time_start": null, "time_end": null, "intent": "why", "sub_queries": ["why JWT?"]}\n```'
    plan = QueryDecomposer(llm=_LLM(reply)).decompose("why do we use JWT?")
    assert plan.intent == "why"
    assert plan.entities == ["JWT"]
    assert not any("LLM" in w for w in plan.warnings)


def test_unparseable_reply_still_falls_back_with_warning():
    plan = QueryDecomposer(llm=_LLM("no json here at all")).decompose("why do we use JWT?")
    assert any("LLM" in w or "decomposition" in w.lower() or "JSON" in w for w in plan.warnings)
