"""tests/test_wave1_B.py — Wave 1 Track B regression tests (audit F06, F07, CODE_MODEL).

All tests are offline: Ollama is emulated with httpx.MockTransport and Gemini
with a tiny fake client.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest

from config.settings import settings, Settings
from embedding.models import (
    OllamaEmbeddingClient,
    EmbeddingIdentity,
    EmbeddingUnavailableError,
    InvalidEmbeddingError,
    EmbeddingIdentityMismatchError,
    EmbeddingBatchError,
    validate_vector,
    code_model_is_distinct,
    PROSE_MODEL,
    CODE_MODEL,
)
from embedding.embedding_cache import EmbeddingCache
from embedding.embedder import DualEmbedder
from retrieval.query_embedder import QueryEmbedder

GOOD = [0.25] * 768


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeOllama:
    """Configurable Ollama emulation.  ``embed_fn(texts) -> list|httpx.Response``."""

    def __init__(self, models=("nomic-embed-text:latest",), embed_fn=None, down=False):
        self.models = list(models)
        self.embed_fn = embed_fn
        self.down = down
        self.embed_calls: list[list[str]] = []
        self.hosts: list[str] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            self.hosts.append(request.url.host)
            if self.down:
                raise httpx.ConnectError("connection refused", request=request)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
            if request.url.path == "/api/embed":
                texts = json.loads(request.content)["input"]
                self.embed_calls.append(texts)
                out = self.embed_fn(texts) if self.embed_fn else [GOOD for _ in texts]
                if isinstance(out, httpx.Response):
                    return out
                return httpx.Response(200, json={"embeddings": out})
            return httpx.Response(404)

        return httpx.Client(transport=httpx.MockTransport(handler))


class FakeGemini:
    def __init__(self, fail_times=0, fail_msg="429 RESOURCE_EXHAUSTED", values=None, always_fail=False):
        self.fail_times = fail_times
        self.fail_msg = fail_msg
        self.values = values
        self.always_fail = always_fail
        self.calls = 0
        self.models = self  # genai.Client().models.embed_content

    def embed_content(self, model, contents):
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_times:
            raise RuntimeError(self.fail_msg)
        n = len(contents) if isinstance(contents, list) else 1
        vals = self.values or ([0.5] * 3072)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(vals)) for _ in range(n)])


def make_client(ollama: FakeOllama | None = None, gemini=None, order=("ollama", "gemini"), sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return OllamaEmbeddingClient(
        http_client=(ollama or FakeOllama(down=True)).client(),
        gemini_client=gemini,
        provider_order=list(order),
        sleep=sleeps.append,
    )


def chunk(i: int, doc_type: str = "commit", text: str | None = None) -> dict:
    return {
        "text": text if text is not None else f"chunk text {i}",
        "metadata": {"chunk_id": f"id_{i}", "doc_type": doc_type},
    }


@pytest.fixture
def cache():
    c = EmbeddingCache(cache_dir=tempfile.mkdtemp())
    yield c
    c.close()


# ----------------------------------------------------------------------
# F06 — typed errors, validation, no zero vectors
# ----------------------------------------------------------------------

class TestF06TypedFailures:
    def test_no_provider_raises_typed_error_not_zeros(self):
        client = make_client(FakeOllama(down=True), gemini=None)
        with pytest.raises(EmbeddingUnavailableError):
            client.embed_single("hello")
        with pytest.raises(EmbeddingUnavailableError):
            client.embed_batch(["a", "b"])

    def test_provider_dies_mid_run_reports_every_unembedded_item(self):
        state = {"n": 0}

        def embed_fn(texts):
            state["n"] += 1
            if state["n"] == 1:
                return [GOOD for _ in texts]
            raise httpx.ConnectError("gone")

        ollama = FakeOllama(embed_fn=embed_fn)
        client = make_client(ollama)
        batch = client.embed_texts([f"t{i}" for i in range(5)], batch_size=2)
        assert [v is not None for v in batch.vectors] == [True, True, False, False, False]
        assert [f.index for f in batch.failures] == [2, 3, 4]
        with pytest.raises(EmbeddingBatchError) as ei:
            batch.require_all()
        assert len(ei.value.failures) == 3

    @pytest.mark.parametrize(
        "bad",
        [
            [0.1] * 767,                       # wrong dimension
            [0.0] * 768,                       # all zero
            [float("nan")] + [0.1] * 767,      # non-finite
            ["x"] * 768,                       # non numeric
        ],
    )
    def test_validate_vector_rejects(self, bad):
        with pytest.raises(InvalidEmbeddingError):
            validate_vector(bad, 768)

    def test_validate_vector_accepts_good(self):
        assert validate_vector(GOOD, 768) == GOOD

    def test_http_zero_and_wrong_dim_reported_per_item(self):
        def embed_fn(texts):
            vecs = [GOOD for _ in texts]
            vecs[1] = [0.0] * 768
            vecs[2] = [0.3] * 100
            return vecs

        client = make_client(FakeOllama(embed_fn=embed_fn))
        batch = client.embed_texts(["a", "b", "c", "d"])
        assert batch.vectors[0] == GOOD and batch.vectors[3] == GOOD
        assert batch.vectors[1] is None and batch.vectors[2] is None
        assert [(f.index, f.error_type) for f in batch.failures] == [
            (1, "InvalidEmbeddingError"), (2, "InvalidEmbeddingError")
        ]

    def test_cardinality_mismatch_retries_singly(self):
        def embed_fn(texts):
            return [GOOD]  # always one vector, regardless of input count

        client = make_client(FakeOllama(embed_fn=embed_fn))
        batch = client.embed_texts(["a", "b", "c"])
        # the 3-item call was short -> each item retried singly -> all succeed
        assert batch.ok and len(batch.vectors) == 3

    def test_cardinality_mismatch_on_singles_is_failure(self):
        client = make_client(FakeOllama(embed_fn=lambda texts: []))
        batch = client.embed_texts(["a", "b"])
        assert batch.vectors == [None, None]
        assert len(batch.failures) == 2

    def test_one_bad_input_does_not_sink_neighbours(self):
        def embed_fn(texts):
            if len(texts) > 1:
                return httpx.Response(500, text="boom")
            if texts[0] == "toolong":
                return httpx.Response(400, text="input length exceeds context")
            return [GOOD]

        client = make_client(FakeOllama(embed_fn=embed_fn))
        batch = client.embed_texts(["ok1", "toolong", "ok2"])
        assert batch.vectors[0] == GOOD and batch.vectors[2] == GOOD
        assert batch.vectors[1] is None
        assert [f.index for f in batch.failures] == [1]

    def test_empty_text_is_a_failure_not_a_zero_vector(self):
        client = make_client(FakeOllama())
        batch = client.embed_texts(["real", "   ", ""])
        assert batch.vectors[0] == GOOD
        assert batch.vectors[1] is None and batch.vectors[2] is None
        assert [f.index for f in batch.failures] == [1, 2]

    def test_gemini_rate_limit_retries_then_succeeds(self):
        sleeps: list = []
        gem = FakeGemini(fail_times=2)
        client = make_client(FakeOllama(down=True), gemini=gem, sleeps=sleeps)
        batch = client.embed_texts(["a"])
        assert batch.ok and batch.identity.provider == "gemini"
        assert len(batch.vectors[0]) == 768  # truncated to the collection dimension
        assert gem.calls == 3 and sleeps == [5, 10]

    def test_gemini_exhausted_is_failure_not_zeros(self):
        client = make_client(FakeOllama(down=True), gemini=FakeGemini(always_fail=True))
        batch = client.embed_texts(["a", "b"])
        assert batch.vectors == [None, None]
        assert len(batch.failures) == 2

    def test_gemini_too_short_vector_rejected(self):
        client = make_client(FakeOllama(down=True), gemini=FakeGemini(values=[0.5] * 100))
        batch = client.embed_texts(["a"])
        assert batch.vectors == [None] and batch.failures[0].error_type == "InvalidEmbeddingError"


class TestF06Cache:
    def test_failures_are_not_cached_and_retry_recovers(self, cache):
        flaky = {"fail": True}

        def embed_fn(texts):
            if flaky["fail"]:
                return [[0.0] * 768 for _ in texts]   # provider returns garbage
            return [GOOD for _ in texts]

        emb = DualEmbedder(client=make_client(FakeOllama(embed_fn=embed_fn)), cache=cache)
        with pytest.raises(EmbeddingBatchError):
            emb.embed_chunks([chunk(0), chunk(1)], show_progress=False)
        assert cache.size == 0

        flaky["fail"] = False
        out = emb.embed_chunks([chunk(0), chunk(1)], show_progress=False)
        assert [c.semantic_vector for c in out] == [GOOD, GOOD]
        assert cache.size == 2

    def test_cache_hit_avoids_provider_call(self, cache):
        ollama = FakeOllama()
        emb = DualEmbedder(client=make_client(ollama), cache=cache)
        emb.embed_chunks([chunk(0)], show_progress=False)
        calls = len(ollama.embed_calls)
        report = emb.embed_chunks_report([chunk(0)], show_progress=False)
        assert len(ollama.embed_calls) == calls and report.cache_hits == 1

    def test_legacy_zero_cache_entry_is_never_reused(self, cache):
        """A zero vector cached by the old code under the requested model name."""
        cache.put("chunk text 0", PROSE_MODEL, [0.0] * 768)   # legacy key namespace
        ollama = FakeOllama()
        emb = DualEmbedder(client=make_client(ollama), cache=cache)
        out = emb.embed_chunks([chunk(0)], show_progress=False)
        assert out[0].semantic_vector == GOOD
        assert len(ollama.embed_calls) == 1

    def test_poisoned_identity_entry_is_a_miss(self, cache):
        ident = EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        key = cache._key("x", ident)
        cache._cache[key] = {"identity": ident.to_dict(), "vector": [0.0] * 768}
        assert cache.get("x", ident) is None
        cache._cache[key] = {
            "identity": EmbeddingIdentity("gemini", "g", 768).to_dict(), "vector": GOOD,
        }
        assert cache.get("x", ident) is None

    def test_put_rejects_invalid_vector_for_identity(self, cache):
        ident = EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        for bad in ([0.0] * 768, [0.1] * 5, [float("inf")] * 768):
            with pytest.raises(InvalidEmbeddingError):
                cache.put("x", ident, bad)
        assert cache.size == 0


class TestF06Embedder:
    def test_partial_failure_reported_per_chunk(self, cache):
        def embed_fn(texts):
            if len(texts) > 1:
                return httpx.Response(500)
            return [[0.0] * 768] if texts[0] == "chunk text 1" else [GOOD]

        emb = DualEmbedder(client=make_client(FakeOllama(embed_fn=embed_fn)), cache=cache)
        chunks = [chunk(0), chunk(1), chunk(2)]
        report = emb.embed_chunks_report(chunks, show_progress=False)
        assert [c.chunk_id for c in report.chunks] == ["id_0", "id_2"]
        assert [(f.chunk_id, f.vector_kind) for f in report.failures] == [("id_1", "semantic")]
        assert not report.ok
        with pytest.raises(EmbeddingBatchError) as ei:
            emb.embed_chunks(chunks, show_progress=False)
        assert "id_1" in str(ei.value)
        partial = emb.embed_chunks(chunks, show_progress=False, allow_partial=True)
        assert [c.chunk_id for c in partial] == ["id_0", "id_2"]

    def test_total_failure_raises_unavailable(self, cache):
        emb = DualEmbedder(client=make_client(FakeOllama(down=True)), cache=cache)
        with pytest.raises(EmbeddingUnavailableError):
            emb.embed_chunks([chunk(0)], show_progress=False)
        assert cache.size == 0


# ----------------------------------------------------------------------
# F07 — identity
# ----------------------------------------------------------------------

class TestF07Identity:
    def test_result_records_actual_identity(self):
        batch = make_client(FakeOllama()).embed_texts(["a"])
        assert batch.identity == EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        assert batch.identity.to_dict() == {
            "provider": "ollama", "model": "nomic-embed-text", "dim": 768, "version": ""
        }
        assert EmbeddingIdentity.from_dict(batch.identity.to_dict()) == batch.identity

    def test_fallback_to_gemini_is_visible_in_identity(self):
        client = make_client(FakeOllama(down=True), gemini=FakeGemini())
        batch = client.embed_texts(["a"])
        assert batch.identity.provider == "gemini"
        assert batch.identity.model == settings.embedding_gemini_model
        assert batch.identity != EmbeddingIdentity("ollama", "nomic-embed-text", 768)

    def test_cache_key_uses_actual_provider_not_requested_model(self, cache):
        # Same text, same requested model, different providers -> separate entries.
        emb_o = DualEmbedder(client=make_client(FakeOllama()), cache=cache)
        emb_o.embed_chunks([chunk(0)], show_progress=False)
        gem = FakeGemini(values=[0.9] * 3072)
        emb_g = DualEmbedder(
            client=make_client(FakeOllama(down=True), gemini=gem), cache=cache
        )
        out = emb_g.embed_chunks([chunk(0)], show_progress=False)
        assert out[0].semantic_vector == [0.9] * 768        # NOT the Ollama cache hit
        assert out[0].semantic_identity.provider == "gemini"
        assert gem.calls == 1 and cache.size == 2

    def test_embedder_pins_identity_no_silent_provider_switch(self, cache):
        """After pinning Ollama, an Ollama outage must NOT fall through to Gemini."""
        state = {"down": False}

        def embed_fn(texts):
            if state["down"]:
                raise httpx.ConnectError("down")
            return [GOOD for _ in texts]

        gem = FakeGemini()
        emb = DualEmbedder(
            client=make_client(FakeOllama(embed_fn=embed_fn), gemini=gem), cache=cache
        )
        emb.embed_chunks([chunk(0)], show_progress=False)
        assert emb.semantic_identity.provider == "ollama"
        state["down"] = True
        with pytest.raises(EmbeddingBatchError):
            emb.embed_chunks([chunk(1)], show_progress=False)
        assert gem.calls == 0

    def test_resolve_with_expected_identity_refuses_other_provider(self):
        expected = EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        client = make_client(FakeOllama(down=True), gemini=FakeGemini())
        with pytest.raises(EmbeddingIdentityMismatchError):
            client.resolve_identity(PROSE_MODEL, expected=expected)

    def test_resolve_with_expected_identity_picks_matching_provider(self):
        expected = EmbeddingIdentity("gemini", settings.embedding_gemini_model, 768)
        client = make_client(FakeOllama(), gemini=FakeGemini())   # Ollama is up too
        assert client.resolve_identity(PROSE_MODEL, expected=expected) == expected

    def test_model_name_latest_tag_is_normalised(self):
        a = EmbeddingIdentity("ollama", "nomic-embed-text:latest", 768)
        b = EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        assert a.matches(b) and a.key == b.key

    def test_different_dim_or_model_do_not_match(self):
        a = EmbeddingIdentity("ollama", "nomic-embed-text", 768)
        assert not a.matches(EmbeddingIdentity("ollama", "nomic-embed-text", 1024))
        assert not a.matches(EmbeddingIdentity("ollama", "other", 768))
        assert not a.matches(EmbeddingIdentity("gemini", "nomic-embed-text", 768))

    def test_embedded_chunks_carry_identity(self, cache):
        emb = DualEmbedder(client=make_client(FakeOllama()), cache=cache)
        out = emb.embed_chunks([chunk(0)], show_progress=False)
        assert out[0].semantic_identity == EmbeddingIdentity("ollama", "nomic-embed-text", 768)


class TestF07QueryEmbedder:
    def test_query_embedder_refuses_mismatch_with_expected_identity(self):
        client = make_client(FakeOllama(down=True), gemini=FakeGemini())
        qe = QueryEmbedder(
            client=client,
            expected_identity=EmbeddingIdentity("ollama", "nomic-embed-text", 768),
        )
        with pytest.raises(EmbeddingIdentityMismatchError):
            qe.embed("query")

    def test_query_embedder_uses_collection_provider_even_if_other_is_up(self):
        expected = {"provider": "gemini", "model": settings.embedding_gemini_model, "dim": 768}
        gem = FakeGemini(values=[0.7] * 3072)
        qe = QueryEmbedder(client=make_client(FakeOllama(), gemini=gem), expected_identity=expected)
        assert qe.embed("query") == [0.7] * 768
        assert qe.identity.provider == "gemini"

    def test_check_collection_identity(self):
        qe = QueryEmbedder(client=make_client(FakeOllama()))
        ident = qe.identity
        qe.check_collection_identity(ident)
        qe.check_collection_identity(ident.to_dict())
        with pytest.raises(EmbeddingIdentityMismatchError):
            qe.check_collection_identity(EmbeddingIdentity("gemini", "g", 768))
        with pytest.raises(EmbeddingIdentityMismatchError):
            qe.check_collection_identity(None)

    def test_query_embed_failure_raises_not_zero_vector(self):
        qe = QueryEmbedder(client=make_client(FakeOllama(down=True)))
        with pytest.raises(EmbeddingUnavailableError):
            qe.embed("query")

    def test_query_embed_invalid_vector_raises(self):
        qe = QueryEmbedder(
            client=make_client(FakeOllama(embed_fn=lambda t: [[0.0] * 768 for _ in t]))
        )
        with pytest.raises(EmbeddingBatchError):
            qe.embed("query")


class TestOllamaBaseUrl:
    def test_default_base_url_comes_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "ollama_base_url", "http://embed-host:9999/")
        client = OllamaEmbeddingClient(http_client=FakeOllama().client())
        assert client.base_url == "http://embed-host:9999"

    def test_explicit_base_url_wins_and_is_used_for_requests(self):
        ollama = FakeOllama()
        client = OllamaEmbeddingClient(
            base_url="http://gpu-box:11434",
            http_client=ollama.client(),
            provider_order=["ollama"],
        )
        client.embed_texts(["a"])
        assert set(ollama.hosts) == {"gpu-box"}

    def test_query_embedder_and_dual_embedder_honor_setting(self, monkeypatch, cache):
        monkeypatch.setattr(settings, "ollama_base_url", "http://from-settings:1234")
        assert QueryEmbedder().client.base_url == "http://from-settings:1234"
        assert DualEmbedder(cache=cache).client.base_url == "http://from-settings:1234"


# ----------------------------------------------------------------------
# CODE_MODEL
# ----------------------------------------------------------------------

class TestCodeModel:
    def test_default_code_model_is_explicitly_the_prose_model(self):
        s = Settings(_env_file=None)
        assert s.embedding_code_model == s.embedding_prose_model == "nomic-embed-text"
        assert CODE_MODEL == PROSE_MODEL
        assert code_model_is_distinct() is False

    def test_code_model_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_CODE_MODEL", "my-code-embed")
        monkeypatch.setenv("EMBEDDING_CODE_DIM", "1024")
        s = Settings(_env_file=None)
        assert s.embedding_code_model == "my-code-embed" and s.embedding_code_dim == 1024

    def test_no_dual_vectors_when_models_are_identical(self, cache):
        ollama = FakeOllama()
        emb = DualEmbedder(client=make_client(ollama), cache=cache)
        assert emb.dual_vectors_enabled is False
        report = emb.embed_chunks_report([chunk(0, "diff"), chunk(1, "code")], show_progress=False)
        assert all(c.code_vector is None for c in report.chunks)
        assert report.dual_vectors is False and report.code_identity is None
        # one embedding per chunk, not two
        assert sum(len(c) for c in ollama.embed_calls) == 2

    def test_dual_vectors_only_for_code_chunks_when_models_differ(self, cache):
        ollama = FakeOllama(models=("nomic-embed-text:latest", "code-embed:latest"))
        emb = DualEmbedder(client=make_client(ollama), cache=cache, code_model="code-embed")
        assert emb.dual_vectors_enabled is True
        report = emb.embed_chunks_report([chunk(0, "commit"), chunk(1, "diff")], show_progress=False)
        by_id = {c.chunk_id: c for c in report.chunks}
        assert by_id["id_0"].code_vector is None
        assert by_id["id_1"].code_vector is not None
        assert by_id["id_1"].code_identity.model == "code-embed"
        assert report.code_identity.model == "code-embed"

    def test_distinct_code_model_not_served_fails_loudly(self, cache):
        ollama = FakeOllama(models=("nomic-embed-text:latest",))   # no code-embed model
        emb = DualEmbedder(client=make_client(ollama), cache=cache, code_model="code-embed")
        with pytest.raises(EmbeddingUnavailableError):
            emb.embed_chunks([chunk(0, "diff")], show_progress=False)
