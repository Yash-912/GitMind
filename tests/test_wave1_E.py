"""tests/test_wave1_E.py — Regression tests for Track E (F27-F30).

Everything here runs offline with fakes: no ragas, mlflow, Gemini or Qdrant.
Real RAGAS / MLflow execution is NOT exercised by this file.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.custom_metrics import (
    EntityConsistencyMetric,
    MultiHopCoverageMetric,
    TemporalAccuracyMetric,
)
from evaluation.dataset_builder import NO_EVIDENCE_ANSWER, DatasetBuilder
from evaluation.mlflow_logger import MLflowLogger
from evaluation.ragas_evaluator import (
    RAGAS_METRIC_NAMES,
    EvalResult,
    RagasBackend,
    RagasConfig,
    RagasEvaluator,
    RagasUnavailableError,
)
from scripts.run_eval import run_configurations


# --------------------------------------------------------------------- F28

CONTEXT = "The release 2.0 shipped the Redis cache in 2021. Alice Smith removed the legacy parser in 2022."


def test_temporal_invented_year_scores_lower_than_correct():
    m = TemporalAccuracyMetric()
    good = m.score_one("Redis cache shipped in 2021.", ["2021-03-01"], CONTEXT)
    bad = m.score_one("Redis cache shipped in 2021 and 1999.", ["2021-03-01"], CONTEXT)
    assert good == 1.0
    assert bad < good


def test_temporal_year_attached_to_wrong_event_is_penalized():
    m = TemporalAccuracyMetric()
    # 2022 exists in the context, but for the parser removal, not the cache.
    swapped = m.score_one("Redis cache shipped in 2022.", ["2021-03-01", "2022-01-01"], CONTEXT)
    right = m.score_one("Redis cache shipped in 2021.", ["2021-03-01", "2022-01-01"], CONTEXT)
    assert swapped < right


def test_temporal_not_applicable_without_years():
    assert TemporalAccuracyMetric().score_one("No dates here.", [], CONTEXT) is None
    assert TemporalAccuracyMetric().score(["No dates here."], [[]]) is None


def test_entity_invented_entities_score_lower():
    m = EntityConsistencyMetric()
    good = m.score_one("Alice removed the parser.", [["alice"]], CONTEXT)
    bad = m.score_one("Alice and Mallory removed the parser.", [["alice"]], CONTEXT)
    assert good == 1.0
    assert bad < good


def test_entity_fully_invented_scores_zero_and_not_applicable_is_none():
    m = EntityConsistencyMetric()
    assert m.score_one("Mallory did it.", [["alice"]], CONTEXT) == 0.0
    assert m.score_one("nothing capitalised here", [["alice"]], CONTEXT) is None


def test_multihop_scored_on_cited_evidence_not_type_diversity():
    m = MultiHopCoverageMetric()
    assert m.score_one(["a", "b"], ["a", "b"]) == 1.0
    assert m.score_one(["a"], ["a", "b"]) == 0.5
    assert m.score_one([], ["a", "b"]) == 0.0
    assert m.score_one(["a"], ["a"]) is None  # single-chunk question: not applicable


def test_multihop_requires_connected_path_when_links_known():
    m = MultiHopCoverageMetric()
    # a and c are both cited but no path joins them -> capped
    assert m.score_one(["a", "c"], ["a", "c"], links=[("a", "x")]) == 0.5
    assert m.score_one(["a", "c"], ["a", "c"], links=[("a", "c")]) == 1.0


def test_aggregate_ignores_not_applicable():
    m = EntityConsistencyMetric()
    assert m.score(["Alice here.", "lowercase only"], [[["alice"]], [[]]]) == 1.0
    assert m.score(["lowercase only"], [[[]]]) is None


# --------------------------------------------------------------------- F27


class _FakeBackend:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def evaluate(self, data):
        self.calls.append(data)
        if self.exc:
            raise self.exc
        return self.result


def _runner(pairs):
    results = [EvalResult(p["question"], "Alice shipped it in 2021.", p["answer"], CONTEXT) for p in pairs]
    meta = {
        "answers": [r.answer for r in results],
        "timestamps": [["2021-03-01"]] * len(results),
        "entity_tags": [[["alice"]]] * len(results),
        "cited_ids": [["a", "b"]] * len(results),
        "latencies": [0.5] * len(results),
    }
    return results, meta


def _dataset(tmp_path: Path, rows: list[dict]) -> str:
    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


class _Cfg:
    gemini_api_key = "k"
    ragas_judge_model = "judge-x"
    ragas_embedding_model = "embed-y"


def test_ragas_config_comes_from_settings_and_env(monkeypatch):
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_EMBEDDING_MODEL", raising=False)
    cfg = RagasConfig.from_settings(_Cfg())
    assert (cfg.judge_model, cfg.embedding_model, cfg.api_key) == ("judge-x", "embed-y", "k")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "env-judge")
    assert RagasConfig.from_settings(_Cfg()).judge_model == "env-judge"


def test_answer_correctness_is_a_selected_metric():
    assert "answer_correctness" in RAGAS_METRIC_NAMES


def test_backend_surfaces_real_exception_not_generic_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)  # force ImportError
    with pytest.raises(RagasUnavailableError) as ei:
        RagasBackend(RagasConfig(api_key="k"))
    assert "ragas is not installed" not in str(ei.value)
    assert "ImportError" in str(ei.value) or "ModuleNotFoundError" in str(ei.value)
    assert ei.value.__cause__ is not None


def test_backend_without_api_key_says_so():
    with pytest.raises(RagasUnavailableError, match="GEMINI_API_KEY"):
        RagasBackend(RagasConfig(api_key=None))


def test_evaluator_runtime_failure_is_raised_not_skipped(tmp_path):
    ds = _dataset(tmp_path, [{"question": "q", "answer": "a", "split": "eval"}])
    boom = RagasUnavailableError("RAGAS evaluation failed: RuntimeError: judge 500")
    ev = RagasEvaluator(config=RagasConfig(), backend=_FakeBackend(exc=boom), pipeline_runner=_runner)
    with pytest.raises(RagasUnavailableError, match="judge 500"):
        ev.evaluate(ds)


def test_evaluator_reports_all_metrics_and_warns_on_nan(tmp_path):
    ds = _dataset(tmp_path, [{"question": "q", "answer": "a", "split": "eval", "required_chunk_ids": ["a", "b"]}])
    raw = {n: 0.9 for n in RAGAS_METRIC_NAMES}
    raw["answer_correctness"] = float("nan")
    ev = RagasEvaluator(config=RagasConfig(), backend=_FakeBackend(raw), pipeline_runner=_runner)
    with pytest.warns(RuntimeWarning, match="answer_correctness"):
        out = ev.evaluate(ds)
    assert out["faithfulness"] == 0.9
    assert out["answer_correctness"] is None  # not silently 0 or dropped
    assert out["multihop_coverage"] == 1.0
    assert out["temporal_accuracy"] == 1.0
    assert any("answer_correctness" in w for w in ev.last_report.warnings)


def test_evaluator_only_uses_requested_split(tmp_path):
    ds = _dataset(
        tmp_path,
        [
            {"question": "q1", "answer": "a", "split": "train"},
            {"question": "q2", "answer": "a", "split": "eval"},
        ],
    )
    backend = _FakeBackend({n: 1.0 for n in RAGAS_METRIC_NAMES})
    RagasEvaluator(config=RagasConfig(), backend=backend, pipeline_runner=_runner).evaluate(ds)
    assert backend.calls[0]["question"] == ["q2"]


# --------------------------------------------------------------------- F29


def _chunk(cid, text, doc_id="", ts="2023-05-01T10:00:00", entities=None):
    return {
        "text": text,
        "metadata": {"chunk_id": cid, "doc_id": doc_id, "timestamp": ts, "entity_tags": entities or []},
    }


def _write_chunks(tmp_path: Path, chunks: list[dict]) -> str:
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return str(p)


def _fake_gen(prompts: list[str] | None = None):
    def gen(prompt: str) -> str:
        if prompts is not None:
            prompts.append(prompt)
        return json.dumps({"question": "What changed about widgets?", "answer": "Widgets were rewritten."})

    return gen


def _corpus(n=20):
    # pairs of chunks share a doc_id so linked clusters exist
    return [_chunk(f"c{i}", f"widget change number {i} rewrote module{i}", doc_id=f"d{i // 2}") for i in range(n)]


def test_seeded_sampling_is_deterministic_and_persisted(tmp_path):
    path = _write_chunks(tmp_path, _corpus())

    def run(seed, name):
        b = DatasetBuilder(generate=_fake_gen())
        pairs = b.build(path, str(tmp_path / name), limit=5, seed=seed, include_adversarial=False)
        return [p.source_chunk_id for p in pairs], str(tmp_path / name)

    a, out_a = run(7, "a.jsonl")
    b_, _ = run(7, "b.jsonl")
    c, _ = run(8, "c.jsonl")
    assert a == b_
    assert a != c
    meta = json.loads(Path(out_a + ".meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7
    assert meta["source_scope"]["n_chunks"] == 20
    assert meta["source_scope"]["chunk_ids_sha256"]
    assert json.loads(Path(out_a).read_text(encoding="utf-8").splitlines()[0])["seed"] == 7


def test_multihop_uses_linked_cluster_not_single_chunk(tmp_path):
    path = _write_chunks(tmp_path, _corpus())
    b = DatasetBuilder(generate=_fake_gen())
    pairs = b.build(path, str(tmp_path / "o.jsonl"), limit=6, seed=1, include_adversarial=False)
    multi = [p for p in pairs if p.qa_type == "multihop"]
    assert multi
    for p in multi:
        assert len(p.required_chunk_ids) >= 2
        assert all(cid in p.context for cid in p.required_chunk_ids)


def test_explicit_links_define_cluster(tmp_path):
    chunks = [_chunk(f"c{i}", f"unique text {i}", doc_id=f"solo{i}") for i in range(4)]
    path = _write_chunks(tmp_path, chunks)
    b = DatasetBuilder(generate=_fake_gen())
    pairs = b.build(path, str(tmp_path / "o.jsonl"), limit=4, seed=0, links=[("c0", "c3")],
                    include_adversarial=False, include_temporal=False)
    multi = [p for p in pairs if p.qa_type == "multihop"]
    assert {frozenset(p.required_chunk_ids) for p in multi} == {frozenset({"c0", "c3"})}
    assert b.rejected["multihop_no_linked_cluster"] == 2  # c1, c2 have no links


def test_unlinked_chunks_yield_no_multihop_question(tmp_path):
    chunks = [_chunk(f"c{i}", f"unique text {i}", doc_id=f"solo{i}") for i in range(3)]
    path = _write_chunks(tmp_path, chunks)
    pairs = DatasetBuilder(generate=_fake_gen()).build(
        path, str(tmp_path / "o.jsonl"), limit=3, include_adversarial=False
    )
    assert not [p for p in pairs if p.qa_type == "multihop"]


def test_links_from_graph_reads_graph_edges(tmp_path):
    db = tmp_path / "g.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE graph_edges (id INTEGER PRIMARY KEY, source_type TEXT, source_id TEXT,"
        " target_type TEXT, target_id TEXT, relation TEXT)"
    )
    conn.execute("INSERT INTO graph_edges(source_type,source_id,target_type,target_id,relation) VALUES"
                 " ('pr','7','issue','3','closes')")
    conn.commit()
    conn.close()
    chunks = [
        {"text": "x", "metadata": {"chunk_id": "p", "graph_node_id": "pr_7"}},
        {"text": "y", "metadata": {"chunk_id": "i", "graph_node_id": "issue_3"}},
    ]
    assert DatasetBuilder.links_from_graph(str(db), chunks) == [("p", "i")]


def test_temporal_prompt_includes_source_timestamp(tmp_path):
    prompts: list[str] = []
    path = _write_chunks(tmp_path, [_chunk("c0", "widget change", ts="2023-05-01T10:00:00")])
    DatasetBuilder(generate=_fake_gen(prompts)).build(
        path, str(tmp_path / "o.jsonl"), limit=1, include_adversarial=False, include_multihop=False
    )
    assert any("2023-05-01T10:00:00" in p and "temporal" in p.lower() for p in prompts)


def test_blank_and_unparseable_generations_are_rejected_not_faked(tmp_path):
    path = _write_chunks(tmp_path, [_chunk("c0", "some very long source text " * 20)])
    for bad in ["not json at all", json.dumps({"question": "q?", "answer": "   "}), json.dumps({"answer": "a"})]:
        b = DatasetBuilder(generate=lambda _p, bad=bad: bad)
        pairs = b.build(path, str(tmp_path / "o.jsonl"), limit=1, include_adversarial=False,
                        include_multihop=False, include_temporal=False)
        assert pairs == []  # no text[:200] fallback answer
        assert sum(b.rejected.values()) == 1


def test_adversarial_question_answerable_elsewhere_is_rejected(tmp_path):
    chunks = [
        _chunk("c0", "payment gateway retry logic was added"),
        _chunk("c1", "kubernetes deployment manifest for staging cluster"),
    ]
    path = _write_chunks(tmp_path, chunks)
    # a chunk other than the source answers the question -> must be rejected
    gen = lambda _p: json.dumps({"question": "Which kubernetes deployment manifest targets the staging cluster?", "answer": "x"})
    b = DatasetBuilder(generate=gen)
    pairs = b.build(path, str(tmp_path / "o.jsonl"), limit=1, seed=0, include_temporal=False,
                    include_multihop=False, max_attempts=2)
    assert not [p for p in pairs if p.qa_type == "adversarial"]
    assert b.rejected["adversarial_answerable_elsewhere"] >= 1
    assert pairs  # the standard pair is still produced


def test_adversarial_unanswerable_question_is_kept_with_canonical_answer(tmp_path):
    chunks = [_chunk("c0", "payment gateway retry logic"), _chunk("c1", "kubernetes manifest staging")]
    path = _write_chunks(tmp_path, chunks)

    def gen(prompt):
        if "NOT answered" in prompt:
            return json.dumps({"question": "Who designed the quantum telemetry exporter?", "answer": "x"})
        return json.dumps({"question": "q?", "answer": "a"})

    pairs = DatasetBuilder(generate=gen).build(
        path, str(tmp_path / "o.jsonl"), limit=2, include_temporal=False, include_multihop=False
    )
    adv = [p for p in pairs if p.qa_type == "adversarial"]
    assert adv and adv[0].answer == NO_EVIDENCE_ANSWER


def test_train_eval_split_has_no_source_leakage(tmp_path):
    path = _write_chunks(tmp_path, _corpus(30))
    pairs = DatasetBuilder(generate=_fake_gen()).build(
        path, str(tmp_path / "o.jsonl"), limit=20, seed=3, eval_fraction=0.3
    )
    splits = {p.split for p in pairs}
    assert splits == {"train", "eval"}
    train_src = {p.source_chunk_id for p in pairs if p.split == "train"}
    eval_src = {p.source_chunk_id for p in pairs if p.split == "eval"}
    assert not train_src & eval_src


def test_existing_dry_run_still_works(tmp_path):
    path = _write_chunks(tmp_path, [_chunk("c1", "hello")])
    pairs = DatasetBuilder(generate=_fake_gen()).build(path, str(tmp_path / "o.jsonl"), limit=1, dry_run=True)
    assert pairs and all(p.dry_run for p in pairs)


# --------------------------------------------------------------------- F30


class _FakeMlflow:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.runs: list[dict] = []
        self.experiment = None

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        self.runs.append({"name": run_name, "params": {}, "metrics": {}, "artifacts": [], "tags": {}})
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def log_params(self, p):
        if self.fail_on == "params":
            raise RuntimeError("tracking server down")
        self.runs[-1]["params"].update(p)

    def log_param(self, k, v):
        self.runs[-1]["params"][k] = v

    def log_metrics(self, m):
        self.runs[-1]["metrics"].update(m)

    def set_tags(self, t):
        self.runs[-1]["tags"].update(t)

    def log_artifact(self, path, artifact_path=None):
        self.runs[-1]["artifacts"].append((artifact_path, Path(path).name))


def test_logger_logs_params_metrics_dataset_and_per_question(tmp_path):
    ds = tmp_path / "qa.jsonl"
    ds.write_text("{}", encoding="utf-8")
    fake = _FakeMlflow()
    lg = MLflowLogger("exp", mlflow_module=fake)
    ok = lg.log_run({"judge_model": "j"}, {"faithfulness": 0.9, "temporal_accuracy": None},
                    run_name="cfg", dataset_path=str(ds), per_question=[{"q": 1}])
    run = fake.runs[0]
    assert ok and run["name"] == "cfg" and fake.experiment == "exp"
    assert run["params"]["judge_model"] == "j"
    assert run["metrics"] == {"faithfulness": 0.9}  # None is not logged as a number
    assert run["params"]["metrics_not_applicable"] == "temporal_accuracy"
    assert ("dataset", "qa.jsonl") in run["artifacts"]
    assert ("results", "per_question_results.json") in run["artifacts"]
    assert lg.status()["runs_logged"] == 1


def test_logger_unavailable_is_visible_not_silent(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.warns(RuntimeWarning, match="MLflow disabled"):
        lg = MLflowLogger("exp")
    assert not lg.enabled and lg.status()["error"]
    with pytest.warns(RuntimeWarning, match="not logged"):
        assert lg.log_run({}, {"m": 1.0}) is False


def test_logger_failure_during_logging_warns_and_reports():
    lg = MLflowLogger("exp", mlflow_module=_FakeMlflow(fail_on="params"))
    with pytest.warns(RuntimeWarning, match="tracking server down"):
        assert lg.log_run({"a": 1}, {"m": 1.0}) is False
    assert "tracking server down" in lg.status()["error"]
    assert lg.status()["runs_logged"] == 0


def test_evaluator_logs_to_mlflow_with_identities_and_latency(tmp_path):
    ds = _dataset(tmp_path, [{"question": "q", "answer": "a", "split": "eval"}])
    fake = _FakeMlflow()
    ev = RagasEvaluator(
        config=RagasConfig(judge_model="jj", embedding_model="ee"),
        backend=_FakeBackend({n: 0.8 for n in RAGAS_METRIC_NAMES}),
        pipeline_runner=_runner,
        logger=MLflowLogger("exp", mlflow_module=fake),
    )
    ev.evaluate(ds, run_name="r1")
    run = fake.runs[0]
    assert run["params"]["judge_model"] == "jj" and run["params"]["embedding_model"] == "ee"
    assert run["metrics"]["faithfulness"] == 0.8
    assert run["metrics"]["latency_total_s"] == 0.5
    assert ev.last_report.mlflow_logged is True


def test_run_eval_logs_one_mlflow_run_per_configuration(tmp_path):
    ds = _dataset(tmp_path, [{"question": "q", "answer": "a", "split": "eval"}])
    fake = _FakeMlflow()
    res = run_configurations(
        {"dense": _runner, "full": _runner},
        ds,
        MLflowLogger("exp", mlflow_module=fake),
        backend=_FakeBackend({n: 0.7 for n in RAGAS_METRIC_NAMES}),
        ragas_config=RagasConfig(),
    )
    assert [r["name"] for r in fake.runs] == ["dense", "full"]
    assert all(r["error"] is None and r["mlflow_logged"] for r in res.values())
    assert fake.runs[0]["params"]["configuration"] == "dense"


def test_run_eval_records_failed_configuration_and_continues(tmp_path):
    ds = _dataset(tmp_path, [{"question": "q", "answer": "a", "split": "eval"}])

    def broken(_pairs):
        raise RuntimeError("retriever down")

    fake = _FakeMlflow()
    res = run_configurations(
        {"broken": broken, "ok": _runner},
        ds,
        MLflowLogger("exp", mlflow_module=fake),
        backend=_FakeBackend({n: 0.7 for n in RAGAS_METRIC_NAMES}),
        ragas_config=RagasConfig(),
    )
    assert "retriever down" in res["broken"]["error"]
    assert res["ok"]["error"] is None
    assert [r["name"] for r in fake.runs] == ["ok"]
