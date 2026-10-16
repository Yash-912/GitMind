"""Wave 2 Track D4: API layer, jobs, readiness, scope enforcement. Offline: fake
retrieval service and fake LLM; subprocess jobs run tiny `python -c` steps."""
from __future__ import annotations

import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.jobs import JobRunner, JobStore, QueueFullError, default_steps
from api.main import create_app
from config.settings import settings
from embedding.models import EmbeddingUnavailableError
from generation.llm_client import LLMUnavailableError
from indexing.generation import GenerationManager

REPO = "owner/repo"
PY = sys.executable


# ------------------------------------------------------------------ fakes --


def cand(cid: str, score: float = 0.5, **meta):
    base = {
        "repo": REPO, "doc_type": "commit", "doc_id": f"sha-{cid}",
        "timestamp": "2024-01-02T00:00:00+00:00", "author": "ann",
        "commit_sha": f"sha-{cid}", "source_path": "auth.py",
        "line_start": 3, "line_end": 9,
    }
    base.update(meta)
    return SimpleNamespace(chunk_id=cid, score=score, source="dense", text=f"text {cid}", metadata=base)


class _FakeQdrant:
    def __init__(self, service):
        self.service, self.up = service, True

    def count(self):
        if not self.up:
            raise ConnectionError("qdrant down")
        return 3


class FakeService:
    def __init__(self, gen, *, candidates=None, included=None, error=None, gate=None):
        self.gen = gen
        self.candidates = candidates if candidates is not None else [cand("c1"), cand("c2"), cand("c3")]
        self.included = included if included is not None else ["c1", "c2"]
        self.error = error
        self.gate = gate
        self.calls: list = []
        self.closed = False
        self.reranker_available = True
        self.startup_warnings: list[str] = []
        # Mirrors the real service: Qdrant is reached through retriever.qdrant.
        self.retriever = SimpleNamespace(qdrant=_FakeQdrant(self))

    @property
    def qdrant_reachable(self) -> bool:
        return self.retriever.qdrant.up

    def retrieve(self, question, scope, top_k=12, limit=40):
        self.calls.append((question, scope, top_k, limit))
        if self.gate is not None:
            self.gate.wait(10)
        if self.error:
            raise self.error
        assembled = SimpleNamespace(
            text="\n---\n".join(f"[chunk:{i}] text" for i in self.included),
            included_ids=list(self.included),
        )
        return SimpleNamespace(
            plan=None, candidates=self.candidates, assembled=assembled, graph_paths=[],
            warnings=["fake warning"], reranker_available=True,
            retrieved_ids=[c.chunk_id for c in self.candidates],
        )

    def close(self):
        self.closed = True


class FakeLLM:
    providers = ["gemini", "ollama"]

    def __init__(self, reply=None):
        self.reply = reply
        self.prompts: list[str] = []
        self.deadlines: list = []
        self.closed = False

    def generate(self, prompt, *, deadline_s=None):
        self.prompts.append(prompt)
        self.deadlines.append(deadline_s)
        out = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out, model="fake-model", provider="fake")

    def close(self):
        self.closed = True


DIRECT_OK = json.dumps({"answer": "Because of JWT.", "confidence": "high", "citations": ["c1"]})


def publish(root, repo=REPO, chunks=3, *, commit="abc123"):
    mgr = GenerationManager(root, repo)
    gen = mgr.new_generation(commit=commit, embedding={"provider": "fake", "dim": 4})
    with gen.chunks_path.open("w", encoding="utf-8") as fh:
        for i in range(chunks):
            fh.write(json.dumps({"text": f"t{i}", "metadata": {"chunk_id": f"c{i}"}}) + "\n")
    gen.mark_ready(chunk_count=chunks)
    mgr.publish(gen)
    return gen


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "generations_dir", str(tmp_path / "data" / "generations"))
    monkeypatch.setattr(settings, "jobs_db_path", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(settings, "allowed_repos", [])
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "query_timeout_s", 50.0)
    return SimpleNamespace(root=str(tmp_path / "data" / "generations"), tmp=tmp_path)


@pytest.fixture
def make_client(env):
    created = []

    def _make(*, reply=DIRECT_OK, service_kwargs=None, steps=None, publish_repo=True, chunks=3):
        services: list[FakeService] = []
        llm = FakeLLM(reply)
        if publish_repo:
            publish(env.root, chunks=chunks)

        def factory(gen):
            svc = FakeService(gen, **(service_kwargs or {}))
            services.append(svc)
            return svc

        app = create_app(
            service_factory=factory, llm_factory=lambda: llm, steps=steps,
            jobs_db=env.tmp / "jobs.db", coverage_fn=lambda g: True,
        )
        client = TestClient(app)
        client.__enter__()
        created.append(client)
        client.services, client.llm, client.app_ = services, llm, app
        return client

    yield _make
    for c in created:
        c.__exit__(None, None, None)


def ask(client, **over):
    body = {"repo": REPO, "query": "why jwt?", "mode": "direct"}
    body.update(over)
    return client.post("/api/v1/query", json=body)


# -------------------------------------------------------------- 422 / input --


@pytest.mark.parametrize(
    "body",
    [
        {"query": "why jwt?"},                                   # repo missing
        {"repo": "", "query": "why jwt?"},
        {"repo": "   ", "query": "why jwt?"},
        {"repo": REPO},                                          # query missing
        {"repo": REPO, "query": ""},
        {"repo": REPO, "query": "     "},
        {"repo": REPO, "query": "ab"},
        {"repo": REPO, "query": "why jwt?", "mode": "bogus"},
        {"repo": REPO, "query": "why jwt?", "kind": "weird"},
        {"repo": REPO, "query": "why jwt?", "time_start": "2024-02-01", "time_end": "2024-01-01"},
        {"repo": REPO, "query": "why jwt?", "time_start": "not-a-date"},
        {"repo": REPO, "query": "why jwt?", "top_k": 0},
    ],
)
def test_invalid_query_input_is_422_before_any_work(make_client, body):
    c = make_client()
    r = c.post("/api/v1/query", json=body)
    assert r.status_code == 422
    assert c.services[0].calls == [] and c.llm.prompts == []


def test_query_is_trimmed(make_client):
    c = make_client()
    assert ask(c, query="   why jwt?   ").json()["query"] == "why jwt?"
    assert c.services[0].calls[0][0] == "why jwt?"


def test_negative_max_commits_is_422(make_client):
    c = make_client()
    assert c.post("/api/v1/ingest", json={"github_repo": REPO, "max_commits": -1}).status_code == 422


def test_scope_filters_reach_retrieval(make_client):
    c = make_client()
    r = ask(c, authors=["ann"], doc_types=["commit"], modules=["auth"], kind="history",
            time_start="2024-01-01T00:00:00Z", time_end="2024-12-31T00:00:00Z")
    assert r.status_code == 200
    scope = c.services[0].calls[0][1]
    assert scope.repo == REPO and scope.authors == ("ann",) and scope.doc_types == ("commit",)
    assert scope.modules == ("auth",) and scope.kind == "history" and scope.time_start is not None


# -------------------------------------------------------- status mapping --


def test_answered_is_200_with_three_separate_source_lists(make_client):
    c = make_client()
    r = ask(c)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["answer"] == "Because of JWT."
    assert body["data"]["confidence"] == "high"
    assert [s["chunk_id"] for s in body["retrieved"]] == ["c1", "c2", "c3"]
    assert [s["chunk_id"] for s in body["included"]] == ["c1", "c2"]
    assert [s["chunk_id"] for s in body["cited"]] == ["c1"]
    loc = body["cited"][0]
    assert (loc["repo"], loc["doc_type"], loc["doc_id"], loc["author"]) == (REPO, "commit", "sha-c1", "ann")
    assert (loc["commit_sha"], loc["path"], loc["line_start"], loc["line_end"]) == ("sha-c1", "auth.py", 3, 9)
    assert loc["timestamp"].startswith("2024-01-02")
    assert body["warnings"] == ["fake warning"] and body["generation_id"]


def test_citing_unincluded_chunk_is_invalid_output_502(make_client):
    bad = json.dumps({"answer": "x", "confidence": "low", "citations": ["c3"]})  # c3 retrieved, not shown
    c = make_client(reply=bad)
    r = ask(c)
    assert r.status_code == 502
    assert r.json()["status"] == "invalid_output" and r.json()["cited"] == []


def test_garbage_model_output_is_502(make_client):
    r = ask(make_client(reply="not json at all"))
    assert r.status_code == 502 and r.json()["status"] == "invalid_output"


def test_no_included_evidence_is_200_insufficient_without_llm_call(make_client):
    c = make_client(service_kwargs={"candidates": [], "included": []})
    r = ask(c)
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_evidence" and r.json()["answer"]
    assert c.llm.prompts == []


def test_provider_outage_is_503_with_error_body(make_client):
    c = make_client(reply=LLMUnavailableError(["gemini: down", "ollama: down"]))
    r = ask(c)
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "provider_unavailable" and "down" in body["error"] and body["answer"] == ""


def test_embedding_outage_is_503_not_500(make_client):
    c = make_client(service_kwargs={"error": EmbeddingUnavailableError("ollama down")})
    r = ask(c)
    assert r.status_code == 503 and r.json()["status"] == "embedding_unavailable"


def test_unknown_repo_is_503_not_ready(make_client):
    c = make_client()
    r = ask(c, repo="other/corpus")
    assert r.status_code == 503 and r.json()["status"] == "index_not_ready"


def test_llm_receives_a_bounded_deadline(make_client):
    c = make_client()
    ask(c)
    assert c.llm.deadlines and all(d is not None and 0 < d <= 50.0 for d in c.llm.deadlines)


def test_exhausted_deadline_stops_generation(make_client, monkeypatch):
    monkeypatch.setattr(settings, "query_timeout_s", 0.0)
    c = make_client()
    r = ask(c)
    assert r.status_code in (503, 504)
    assert c.llm.prompts == []


# ------------------------------------------------------------ memo / blame --

MEMO = {"decision": "Use JWT", "context": "ctx", "rationale": "stateless", "evidence": ["c1"]}


def test_memo_evidence_must_be_in_included(make_client):
    ok = ask(make_client(reply=json.dumps(MEMO)), mode="memo")
    assert ok.status_code == 200
    assert ok.json()["cited"][0]["chunk_id"] == "c1" and ok.json()["data"]["decision"] == "Use JWT"

    bad = dict(MEMO, evidence=["c1", "c3"])           # c3 never shown to the model
    r = ask(make_client(reply=json.dumps(bad)), mode="memo")
    assert r.status_code == 502 and r.json()["status"] == "invalid_output"

    none = dict(MEMO, evidence=[])
    assert ask(make_client(reply=json.dumps(none)), mode="memo").status_code == 502


def test_memo_non_json_is_502_and_outage_is_503(make_client):
    assert ask(make_client(reply="prose"), mode="memo").status_code == 502
    assert ask(make_client(reply=LLMUnavailableError(["x"])), mode="memo").status_code == 503


@pytest.mark.parametrize("mode", ["blame", "risk"])
def test_blame_and_risk_compute_statistics_from_stored_records(make_client, mode):
    c = make_client(reply="narration")
    r = ask(c, mode=mode)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "narration"
    assert body["statistics"] is not None and body["data"]["statistics"] == body["statistics"]


# ------------------------------------------------------------- readiness --


def test_health_is_liveness_only_even_when_nothing_is_indexed(make_client):
    c = make_client(publish_repo=False)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


def test_ready_reports_generation_details(make_client):
    c = make_client()
    r = c.get("/ready")
    assert r.status_code == 200
    body = r.json()
    repo = body["repos"][0]
    assert body["status"] == "ready" and repo["ready"] and repo["repo"] == REPO
    assert repo["commit"] == "abc123" and repo["chunk_count"] == 3 and repo["generation_id"]
    assert repo["embedding"] == {"provider": "fake", "dim": 4}
    assert repo["qdrant_reachable"] is True and repo["reranker_available"] is True
    assert repo["generation_providers"] == ["gemini", "ollama"] and repo["degraded"] == []


def test_ready_lists_degraded_components_explicitly(make_client):
    c = make_client()
    svc = c.services[0]
    svc.retriever.qdrant.up, svc.reranker_available = False, False
    r = c.get("/ready")
    assert r.status_code == 503
    repo = r.json()["repos"][0]
    assert not repo["ready"] and {"qdrant_unreachable", "reranker_unavailable"} <= set(repo["degraded"])
    assert any("qdrant_unreachable" in d for d in r.json()["degraded"])


def test_empty_corpus_is_not_ready(make_client):
    c = make_client(chunks=0)
    r = c.get("/ready")
    assert r.status_code == 503 and "empty_corpus" in r.json()["repos"][0]["degraded"]


def test_load_failure_is_visible_in_ready(env, monkeypatch):
    publish(env.root)

    def boom(gen):
        raise RuntimeError("qdrant collection missing")

    app = create_app(service_factory=boom, llm_factory=lambda: FakeLLM(DIRECT_OK),
                     jobs_db=env.tmp / "jobs.db")
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert any("load_failed" in d for d in r.json()["repos"][0]["degraded"])
        assert ask(c).status_code == 503


# ------------------------------------------------------- resources / swap --


def test_resources_load_once_not_per_request(make_client):
    c = make_client()
    for _ in range(3):
        assert ask(c).status_code == 200
    assert len(c.services) == 1 and len(c.services[0].calls) == 3


def test_generation_swap_during_request_is_consistent(make_client, env):
    gate = threading.Event()
    c = make_client(service_kwargs={"gate": gate})
    registry = c.app_.state.registry
    first = c.services[0]
    old_id = first.gen.generation_id
    result = {}

    def slow():
        result["r"] = ask(c)

    t = threading.Thread(target=slow)
    t.start()
    deadline = time.time() + 5
    while not first.calls and time.time() < deadline:
        time.sleep(0.01)
    assert first.calls, "request never reached retrieval"

    new_gen = publish(env.root, commit="def456")
    assert registry.refresh(REPO) is True
    assert len(c.services) == 2
    assert first.closed is False           # still serving the in-flight request
    gate.set()
    t.join(10)
    assert result["r"].status_code == 200
    assert result["r"].json()["generation_id"] == old_id
    assert first.closed is True            # released after its last lease
    assert ask(c).json()["generation_id"] == new_gen.generation_id
    assert c.services[1].closed is False


def test_externally_published_generation_is_picked_up(make_client, env):
    c = make_client()
    gen2 = publish(env.root, commit="zzz")
    assert ask(c).json()["generation_id"] == gen2.generation_id
    assert len(c.services) == 2 and c.services[0].closed


def test_resources_are_closed_on_shutdown(env):
    publish(env.root)
    services, llm = [], FakeLLM(DIRECT_OK)

    def factory(gen):
        services.append(FakeService(gen))
        return services[-1]

    app = create_app(service_factory=factory, llm_factory=lambda: llm, jobs_db=env.tmp / "jobs.db")
    with TestClient(app):
        pass
    assert services[0].closed and llm.closed


# ------------------------------------------------------ allowed / API key --


def test_allowed_repos_enforced_for_query_and_ingest(make_client, monkeypatch):
    c = make_client()
    monkeypatch.setattr(settings, "allowed_repos", [REPO])
    assert ask(c).status_code == 200
    denied = ask(c, repo="other/corpus")
    assert denied.status_code == 403 and denied.json()["status"] == "repo_not_allowed"
    r = c.post("/api/v1/ingest", json={"github_repo": "other/corpus"})
    assert r.status_code == 403
    assert c.services[0].calls[-1][1].repo == REPO


def test_allowed_repos_parses_comma_separated_env(monkeypatch):
    from config.settings import Settings

    monkeypatch.setenv("ALLOWED_REPOS", "a/b, c/d")
    assert Settings().allowed_repos == ["a/b", "c/d"]
    monkeypatch.setenv("ALLOWED_REPOS", "")
    assert Settings().allowed_repos == []


def test_api_key_protects_everything_but_liveness(make_client, monkeypatch):
    c = make_client()
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert c.get("/health").status_code == 200
    assert ask(c).status_code == 401
    assert c.get("/ready").status_code == 401
    assert c.get("/api/v1/jobs").status_code == 401
    ok = c.post("/api/v1/query", json={"repo": REPO, "query": "why jwt?"},
                headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    bad = c.post("/api/v1/query", json={"repo": REPO, "query": "why jwt?"},
                 headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401


# ------------------------------------------------------------------ jobs --


def wait_for(store, job_id, states, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = store.get(job_id)
        if job["status"] in states:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stuck in {store.get(job_id)['status']}")


def py_step(code, label="step"):
    return (label, [PY, "-u", "-c", code])


def test_job_state_persists_and_running_jobs_fail_on_restart(tmp_path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    running = store.create(REPO, {})
    queued = store.create(REPO, {})
    done = store.create(REPO, {})
    store.update(running["id"], status="running", started_at="t")
    store.update(done["id"], status="succeeded")
    store.close()

    reopened = JobStore(path)
    assert reopened.get(running["id"])["status"] == "running"      # durable
    assert reopened.mark_interrupted() == 2
    for jid in (running["id"], queued["id"]):
        job = reopened.get(jid)
        assert job["status"] == "failed" and "restart" in job["error"] and job["finished_at"]
    assert reopened.get(done["id"])["status"] == "succeeded"
    reopened.close()


def test_job_runs_steps_in_order_streams_progress_and_publishes(tmp_path):
    store = JobStore(tmp_path / "j.db")
    published = []
    out = tmp_path / "order.txt"
    steps = lambda repo, p: [
        py_step(f"open(r'{out}','a').write('1');print('first done',flush=True)", "ingest"),
        py_step(f"open(r'{out}','a').write('2');print('second done',flush=True)", "chunk"),
    ]
    runner = JobRunner(store, steps=steps, on_success=published.append)
    job = runner.submit(REPO, {})
    final = wait_for(store, job["id"], {"succeeded", "failed"})
    runner.shutdown()
    assert final["status"] == "succeeded" and final["progress"] == "done"
    assert out.read_text() == "12" and published == [REPO]
    assert final["started_at"] and final["finished_at"]
    store.close()


def test_failed_step_marks_job_failed_with_error_tail_and_does_not_publish(tmp_path):
    store = JobStore(tmp_path / "j.db")
    published = []
    steps = lambda r, p: [
        py_step("import sys;print('boom detail',flush=True);sys.exit(3)", "ingest"),
        py_step("print('never')", "chunk"),
    ]
    runner = JobRunner(store, steps=steps, on_success=published.append)
    job = runner.submit(REPO, {})
    final = wait_for(store, job["id"], {"succeeded", "failed"})
    runner.shutdown()
    assert final["status"] == "failed" and "code 3" in final["error"] and "boom detail" in final["error"]
    assert published == []
    store.close()


def test_time_budget_kills_a_runaway_step(tmp_path):
    store = JobStore(tmp_path / "j.db")
    runner = JobRunner(store, steps=lambda r, p: [py_step("import time;time.sleep(60)")], timeout_s=0.5)
    job = runner.submit(REPO, {})
    final = wait_for(store, job["id"], {"succeeded", "failed"})
    runner.shutdown()
    assert final["status"] == "failed" and "time budget" in final["error"]
    store.close()


def test_cancel_running_and_queued_jobs(tmp_path):
    store = JobStore(tmp_path / "j.db")
    runner = JobRunner(store, steps=lambda r, p: [py_step("import time;time.sleep(60)")])
    first = runner.submit(REPO, {})
    second = runner.submit(REPO, {})
    wait_for(store, first["id"], {"running"})
    assert store.get(second["id"])["status"] == "queued"

    assert runner.cancel(second["id"])["status"] == "cancelled"
    runner.cancel(first["id"])
    assert wait_for(store, first["id"], {"cancelled", "failed"})["status"] == "cancelled"
    time.sleep(0.8)   # the cancelled queued job must never start
    assert store.get(second["id"])["status"] == "cancelled"
    assert runner.cancel("missing") is None
    runner.shutdown()
    store.close()


def test_same_repo_jobs_are_serialized_other_repos_are_not_blocked(tmp_path):
    store = JobStore(tmp_path / "j.db")
    runner = JobRunner(store, steps=lambda r, p: [py_step("import time;time.sleep(0.4)")])
    ids = [runner.submit(REPO, {})["id"] for _ in range(3)]
    other = runner.submit("owner/other", {})["id"]
    for jid in ids + [other]:
        assert wait_for(store, jid, {"succeeded", "failed"})["status"] == "succeeded"
    runner.shutdown()
    assert runner.max_concurrent[REPO] == 1
    finished = [store.get(i)["finished_at"] for i in ids]
    started = [store.get(i)["started_at"] for i in ids]
    assert all(started[n + 1] >= finished[n] for n in range(2))
    store.close()


def test_queue_is_bounded(tmp_path):
    store = JobStore(tmp_path / "j.db")
    runner = JobRunner(store, steps=lambda r, p: [py_step("import time;time.sleep(60)")], queue_size=1)
    runner.submit(REPO, {})
    time.sleep(0.5)                       # worker takes the first; queue is empty again
    runner.submit(REPO, {})               # fills the queue
    with pytest.raises(QueueFullError):
        runner.submit(REPO, {})
    runner.shutdown()
    store.close()


def test_default_steps_use_argument_lists_and_unbuffered_scripts():
    steps = default_steps(REPO, {"github_repo": REPO, "repo_path": ".", "max_commits": 5})
    labels = [s[0] for s in steps]
    assert labels == ["ingest", "parse and chunk", "embed and index"]
    ingest, phase2 = steps[0][1], steps[1][1]
    for argv in (ingest, phase2):
        assert isinstance(argv, list) and argv[:2] == [sys.executable, "-u"]
    assert "--github-repo" in ingest and "--repo-path" not in ingest   # "." placeholder dropped
    assert ingest[ingest.index("--max-commits") + 1] == "5"
    assert phase2[-2:] == ["--repo", REPO]
    phase3 = steps[2][1]
    assert callable(phase3)                       # needs phase 2's generation id
    with pytest.raises(RuntimeError, match="GENERATION_ID"):
        phase3({})
    argv = phase3({"GENERATION_ID": "20240101T000000000000Z-abc123"})
    assert argv[argv.index("--generation") + 1] == "20240101T000000000000Z-abc123"
    assert argv[argv.index("--repo") + 1] == REPO


def test_phase2_generation_id_is_passed_to_phase3(tmp_path):
    store = JobStore(tmp_path / "j.db")
    received = tmp_path / "phase3_args.txt"
    steps = lambda r, p: [
        py_step("print('working');print('GENERATION_ID=gen-42',flush=True)", "parse and chunk"),
        (
            "embed and index",
            lambda captured: [PY, "-u", "-c",
                              f"open(r'{received}','w').write('{captured.get('GENERATION_ID')}')"],
        ),
    ]
    runner = JobRunner(store, steps=steps)
    job = runner.submit(REPO, {})
    final = wait_for(store, job["id"], {"succeeded", "failed"})
    runner.shutdown()
    assert final["status"] == "succeeded"
    assert received.read_text() == "gen-42"
    store.close()


def test_missing_generation_id_fails_the_job_before_embedding(tmp_path):
    store = JobStore(tmp_path / "j.db")
    steps = lambda r, p: [
        py_step("print('no id here')", "parse and chunk"),
        ("embed and index", _phase3_argv_for_test()),
    ]
    runner = JobRunner(store, steps=steps)
    job = runner.submit(REPO, {})
    final = wait_for(store, job["id"], {"succeeded", "failed"})
    runner.shutdown()
    assert final["status"] == "failed" and "GENERATION_ID" in final["error"]
    store.close()


def _phase3_argv_for_test():
    from api.jobs import _phase3_argv

    return _phase3_argv(REPO)


def test_generations_dir_must_match_phase2_location(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "generations_dir", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="DATA_DIR/generations"):
        create_app(service_factory=lambda g: FakeService(g), llm_factory=lambda: FakeLLM(DIRECT_OK),
                   jobs_db=tmp_path / "jobs.db")


def test_ingest_endpoint_queues_polls_and_cancels(make_client):
    c = make_client(steps=lambda r, p: [py_step("import time;time.sleep(60)")])
    r = c.post("/api/v1/ingest", json={"github_repo": "Owner/Repo", "max_commits": 0})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["repo"] == "owner/repo" and r.json()["task_id"] == job_id
    deadline = time.time() + 10
    while c.get(f"/api/v1/jobs/{job_id}").json()["status"] != "running" and time.time() < deadline:
        time.sleep(0.05)
    assert c.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
    assert any(j["id"] == job_id for j in c.get("/api/v1/jobs", params={"repo": "owner/repo"}).json())
    assert c.get(f"/api/v1/status/{job_id}").json()["status"] == "running"

    c.post(f"/api/v1/jobs/{job_id}/cancel")
    deadline = time.time() + 10
    while c.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert c.get(f"/api/v1/jobs/{job_id}").json()["status"] == "cancelled"
    assert c.get("/api/v1/jobs/nope").status_code == 404
    assert c.post("/api/v1/jobs/nope/cancel").status_code == 404


def test_successful_job_swaps_in_the_new_generation(make_client, env):
    c = make_client(steps=lambda r, p: [py_step("print('built')")])
    before = ask(c).json()["generation_id"]
    new_gen = publish(env.root, commit="newcommit")          # what run_phase3 --generation does
    job_id = c.post("/api/v1/ingest", json={"github_repo": REPO}).json()["job_id"]
    deadline = time.time() + 15
    while c.get(f"/api/v1/jobs/{job_id}").json()["status"] not in ("succeeded", "failed") \
            and time.time() < deadline:
        time.sleep(0.05)
    assert c.get(f"/api/v1/jobs/{job_id}").json()["status"] == "succeeded"
    assert c.app_.state.registry.current(REPO).generation_id == new_gen.generation_id != before


def test_restart_marks_interrupted_jobs_failed_via_app_lifespan(env):
    store = JobStore(env.tmp / "jobs.db")
    stale = store.create(REPO, {})
    store.update(stale["id"], status="running")
    store.close()
    app = create_app(service_factory=lambda g: FakeService(g), llm_factory=lambda: FakeLLM(DIRECT_OK),
                     jobs_db=env.tmp / "jobs.db")
    with TestClient(app) as c:
        job = c.get(f"/api/v1/jobs/{stale['id']}").json()
        assert job["status"] == "failed" and "restart" in job["error"]


# ------------------------------------------------------------ deployment --

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def _fake_bin(tmp_path, **scripts):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    for name, body in scripts.items():
        f = bindir / name
        f.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8", newline="\n")
        f.chmod(0o755)
    return bindir


def _run_start(tmp_path, env_extra=None, **scripts):
    import os

    bindir = _fake_bin(tmp_path, **scripts)
    env = {**os.environ, "PATH": f"{bindir.as_posix()}:{os.environ['PATH']}", "API_WAIT_S": "3"}
    env.update(env_extra or {})
    return subprocess.run([BASH, "start.sh"], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_start_sh_exits_nonzero_when_a_service_dies(tmp_path):
    r = _run_start(tmp_path, uvicorn="sleep 30", curl="exit 0", streamlit="sleep 1; exit 0")
    assert r.returncode != 0 and "shutting down" in r.stderr


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_start_sh_does_not_start_streamlit_before_api_is_healthy(tmp_path):
    marker = tmp_path / "streamlit_started"
    touch = f"touch {marker.as_posix()}"
    (tmp_path / "never_healthy").mkdir()
    r = _run_start(tmp_path / "never_healthy", uvicorn="sleep 30", curl="exit 1", streamlit=touch)
    assert r.returncode != 0 and "did not become healthy" in r.stderr and not marker.exists()
    (tmp_path / "dies").mkdir()
    r = _run_start(tmp_path / "dies", uvicorn="exit 7", curl="exit 1", streamlit=touch)
    assert r.returncode != 0 and "exited during startup" in r.stderr and not marker.exists()


def test_compose_has_pinned_images_healthchecks_and_health_conditions():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert ":latest" not in text and "qdrant/qdrant:v1.15.0" in text
    assert text.count("healthcheck:") == 3
    assert "condition: service_healthy" in text and text.count("condition: service_healthy") == 2
    commands = [l for l in text.splitlines() if l.strip().startswith("command:")]
    assert commands and not any("--reload" in l for l in commands)
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11.9-slim-bookworm" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "-c requirements.lock" in dockerfile


def test_requirements_are_compatible_release_pins():
    lines = [l.split("#")[0].strip() for l in (ROOT / "requirements.txt").read_text("utf-8").splitlines()]
    reqs = [l for l in lines if l]
    assert reqs and all("~=" in l for l in reqs), [l for l in reqs if "~=" not in l]
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "fastapi==" in lock and "qdrant-client==" in lock


# ------------------------------------------------------------- UI client --


def test_ui_client_normalises_errors_and_builds_payloads():
    import httpx

    from interface.api_client import ApiError, GitMindClient, build_query_payload

    def handler(request):
        if request.url.path == "/api/v1/query":
            body = json.loads(request.content)
            if body["query"] == "outage":
                return httpx.Response(503, json={"status": "provider_unavailable", "error": "llm down"})
            if body["query"] == "bad":
                return httpx.Response(422, json={"detail": [{"loc": ["body", "repo"], "msg": "Field required"}]})
            return httpx.Response(200, json={"status": "answered", "echo": body})
        if request.url.path == "/ready":
            return httpx.Response(503, json={"status": "not_ready", "repos": [], "degraded": []})
        return httpx.Response(404, json={"detail": "Job not found"})

    client = GitMindClient("http://api", "k", transport=httpx.MockTransport(handler))
    payload = build_query_payload(REPO, " why? ", "direct", authors="ann, bob", kind="any", time_start="2024-01-01")
    assert payload["authors"] == ["ann", "bob"] and "kind" not in payload and payload["query"] == "why?"
    assert client.query(payload)["status"] == "answered"
    with pytest.raises(ApiError) as e:
        client.query({**payload, "query": "outage"})
    assert (e.value.status_code, e.value.kind, e.value.message) == (503, "provider_unavailable", "llm down")
    with pytest.raises(ApiError) as e:
        client.query({**payload, "query": "bad"})
    assert e.value.status_code == 422 and "repo" in e.value.message
    assert client.ready()["status"] == "not_ready"          # 503 report is data, not an error
    with pytest.raises(ApiError) as e:
        client.job("nope")
    assert e.value.status_code == 404
