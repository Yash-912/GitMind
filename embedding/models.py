"""models.py — embedding providers (Ollama primary, Gemini optional) with typed errors.

Design rules (audit F06/F07):

* A provider failure is an exception or a per-item failure, never a vector.
  There is no "return zeros" fallback.
* Every vector is validated (dimension, finite, not all zero) and the batch
  cardinality is checked against the number of inputs.
* Every result carries the *actual* ``EmbeddingIdentity`` (provider, model,
  dim) that produced it.  Provider fallback happens only when an identity is
  *resolved* (``resolve_identity``), never silently mid-run: once an identity
  is pinned, ``embed_texts`` uses that provider only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import httpx

from config.settings import settings


# ----------------------------------------------------------------------
# Configuration constants (overridable through config/settings.py)
# ----------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:11434"

# Model names as served by Ollama.  CODE_MODEL defaults to the *same* model as
# PROSE_MODEL: there is currently no distinct code-embedding model, so the
# "code" vector is NOT a code-specific representation unless the operator sets
# EMBEDDING_CODE_MODEL to a different model (see ``code_model_is_distinct``).
PROSE_MODEL = settings.embedding_prose_model
CODE_MODEL = settings.embedding_code_model

PROSE_DIM = settings.embedding_prose_dim
CODE_DIM = settings.embedding_code_dim


def _norm_model(model: str) -> str:
    m = (model or "").strip()
    return m[: -len(":latest")] if m.endswith(":latest") else m


def code_model_is_distinct(
    prose_model: str = PROSE_MODEL, code_model: str = CODE_MODEL
) -> bool:
    """True only when the code model really differs from the prose model."""
    return _norm_model(prose_model) != _norm_model(code_model)


def _default_dim(model: str) -> int:
    if code_model_is_distinct() and _norm_model(model) == _norm_model(CODE_MODEL):
        return CODE_DIM
    return PROSE_DIM


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------

class EmbeddingError(Exception):
    """Base class for all embedding failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """A provider is missing, unreachable, or refused the request.

    ``fatal`` marks transport-level failures after which retrying other items
    against the same provider is pointless.
    """

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class InvalidEmbeddingError(EmbeddingError):
    """A provider returned a vector (or batch) that failed validation."""


class EmbeddingIdentityMismatchError(EmbeddingError):
    """The available embedding identity differs from the expected one."""


class EmbeddingBatchError(EmbeddingError):
    """Some items of a batch failed.  ``batch`` holds the full per-item report."""

    def __init__(self, message: str, batch: "EmbeddingBatch | None" = None) -> None:
        super().__init__(message)
        self.batch = batch
        self.failures = list(batch.failures) if batch is not None else []


# ----------------------------------------------------------------------
# Identity / results
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingIdentity:
    """Which vector space a vector lives in.  Persist this with the index."""
    provider: str
    model: str
    dim: int
    # Provider-reported model revision when known.  Neither Ollama's /api/embed
    # nor the Gemini embed call reports one, so this is "" today.
    version: str = ""

    def matches(self, other: "EmbeddingIdentity") -> bool:
        """Same vector space? (version is compared only if both sides know it)."""
        if (self.provider, _norm_model(self.model), self.dim) != (
            other.provider, _norm_model(other.model), other.dim
        ):
            return False
        return not (self.version and other.version) or self.version == other.version

    @property
    def key(self) -> str:
        v = f"@{self.version}" if self.version else ""
        return f"{self.provider}/{_norm_model(self.model)}{v}/{self.dim}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingIdentity":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            dim=int(data["dim"]),
            version=str(data.get("version", "")),
        )


@dataclass(frozen=True)
class EmbeddingFailure:
    """One input that could not be embedded."""
    index: int
    error_type: str
    message: str


@dataclass
class EmbeddingBatch:
    """Result of ``embed_texts``.  ``vectors[i]`` is None iff input *i* failed."""
    vectors: list[list[float] | None]
    identity: EmbeddingIdentity
    failures: list[EmbeddingFailure] = field(default_factory=list)
    cache_hits: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def require_all(self) -> list[list[float]]:
        """Return all vectors or raise EmbeddingBatchError naming the failures."""
        if self.failures:
            first = self.failures[0]
            raise EmbeddingBatchError(
                f"{len(self.failures)}/{len(self.vectors)} embeddings failed "
                f"(first: item {first.index}: {first.error_type}: {first.message})",
                self,
            )
        return self.vectors  # type: ignore[return-value]


@dataclass
class EmbeddingResult:
    """Result from a single embedding call."""
    vector: list[float]
    model: str
    token_count: int = 0
    identity: EmbeddingIdentity | None = None


def validate_vector(vector, dim: int) -> list[float]:
    """Return *vector* as a list of floats, or raise InvalidEmbeddingError."""
    if not isinstance(vector, (list, tuple)):
        raise InvalidEmbeddingError(f"embedding is {type(vector).__name__}, not a list")
    if len(vector) != dim:
        raise InvalidEmbeddingError(f"embedding has dimension {len(vector)}, expected {dim}")
    try:
        out = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError(f"embedding contains non-numeric values: {exc}") from exc
    if not all(math.isfinite(x) for x in out):
        raise InvalidEmbeddingError("embedding contains NaN or infinite values")
    if not any(x != 0.0 for x in out):
        raise InvalidEmbeddingError("embedding is all zeros")
    return out


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------

class _OllamaProvider:
    name = "ollama"

    def __init__(self, http: httpx.Client, base_url: str) -> None:
        self.http = http
        self.base_url = base_url

    def identity(self, model: str, dim: int) -> EmbeddingIdentity:
        return EmbeddingIdentity("ollama", _norm_model(model), dim)

    def available(self, model: str) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            want = _norm_model(model)
            return any(
                _norm_model(m.get("name", "")) == want
                for m in resp.json().get("models", [])
            )
        except Exception:
            return False

    def reachable(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def embed(self, texts: list[str], model: str, dim: int) -> list:
        try:
            resp = self.http.post(
                f"{self.base_url}/api/embed",
                json={"model": _norm_model(model), "input": texts},
            )
        except httpx.TransportError as exc:
            raise EmbeddingUnavailableError(f"Ollama unreachable: {exc}", fatal=True) from exc
        if resp.status_code != 200:
            raise EmbeddingUnavailableError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()["embeddings"]
        except Exception as exc:
            raise InvalidEmbeddingError(f"malformed Ollama response: {exc}") from exc


class _GeminiProvider:
    name = "gemini"

    def __init__(self, gemini_client, model: str, sleep=time.sleep) -> None:
        self.gemini = gemini_client
        self.model = model
        self._sleep = sleep

    def identity(self, model: str, dim: int) -> EmbeddingIdentity:
        # The requested (Ollama) model name is irrelevant: Gemini serves its own.
        return EmbeddingIdentity("gemini", self.model, dim)

    def available(self, model: str) -> bool:
        return self.gemini is not None

    def reachable(self) -> bool:
        return self.gemini is not None

    def embed(self, texts: list[str], model: str, dim: int) -> list:
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.gemini.models.embed_content(model=self.model, contents=texts)
                embs = resp.embeddings or []
                # Truncation to *dim* (Matryoshka); shorter vectors fail validation.
                return [list(e.values)[:dim] for e in embs]
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                msg = str(exc)
                limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
                if limited and attempt < 4:
                    self._sleep((2 ** attempt) * 5)
                    continue
                break
        raise EmbeddingUnavailableError(
            f"Gemini embedding failed: {last}", fatal=True
        ) from last


class OllamaEmbeddingClient:
    """Embedding client: Ollama first, optional Gemini, explicit identities.

    ``base_url`` defaults to ``settings.ollama_base_url``.
    Provider order comes from ``settings.embedding_provider``:
    ``"auto"`` (Ollama, then Gemini if GEMINI_API_KEY is set), ``"ollama"`` or
    ``"gemini"``.  Fallback only happens in ``resolve_identity``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
        gemini_client=None,
        provider_order: list[str] | None = None,
        sleep=time.sleep,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client = http_client or httpx.Client(timeout=timeout)

        self._gemini = gemini_client
        if self._gemini is None and settings.gemini_api_key:
            try:
                from google import genai
                self._gemini = genai.Client(api_key=settings.gemini_api_key)
            except Exception:
                self._gemini = None

        self._providers = {
            "ollama": _OllamaProvider(self._client, self.base_url),
            "gemini": _GeminiProvider(self._gemini, settings.embedding_gemini_model, sleep),
        }
        if provider_order is None:
            mode = (settings.embedding_provider or "auto").lower()
            provider_order = {"ollama": ["ollama"], "gemini": ["gemini"]}.get(
                mode, ["ollama", "gemini"]
            )
        self.provider_order = provider_order

    # ------------------------------------------------------------------
    # Identity resolution (the only place where provider fallback happens)
    # ------------------------------------------------------------------

    def resolve_identity(
        self,
        model: str = PROSE_MODEL,
        dim: int | None = None,
        expected: EmbeddingIdentity | None = None,
    ) -> EmbeddingIdentity:
        """Pick the provider that will serve *model*.

        With *expected*, only a provider whose identity matches it is
        acceptable; otherwise EmbeddingIdentityMismatchError is raised rather
        than silently using another vector space.
        """
        dim = dim or (expected.dim if expected else _default_dim(model))
        skipped: list[str] = []
        for name in self.provider_order:
            prov = self._providers[name]
            ident = prov.identity(model, dim)
            if expected is not None and not ident.matches(expected):
                skipped.append(ident.key)
                continue
            if prov.available(model):
                return ident
        if expected is not None:
            raise EmbeddingIdentityMismatchError(
                f"expected embedding identity {expected.key} is not available "
                f"(configured providers would give: {skipped or 'nothing'})"
            )
        raise EmbeddingUnavailableError(
            f"no embedding provider available for model {model!r} "
            f"(tried {self.provider_order})"
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        model: str = PROSE_MODEL,
        *,
        dim: int | None = None,
        identity: EmbeddingIdentity | None = None,
        batch_size: int = 32,
    ) -> EmbeddingBatch:
        """Embed *texts* with exactly one provider/model; report per-item failures.

        ``vectors`` always has ``len(texts)`` entries; failed items are None and
        listed in ``failures``.  Raises EmbeddingUnavailableError only when no
        provider can be selected at all.
        """
        dim = dim or (identity.dim if identity else _default_dim(model))
        if identity is None:
            identity = self.resolve_identity(model, dim)
        prov = self._providers.get(identity.provider)
        if prov is None:
            raise EmbeddingUnavailableError(f"unknown embedding provider {identity.provider!r}")

        vectors: list[list[float] | None] = [None] * len(texts)
        failures: list[EmbeddingFailure] = []
        fatal: str | None = None

        def fail(i: int, exc: Exception) -> None:
            failures.append(EmbeddingFailure(i, type(exc).__name__, str(exc)))

        for start in range(0, len(texts), batch_size):
            idxs = list(range(start, min(start + batch_size, len(texts))))
            if fatal is not None:
                for i in idxs:
                    fail(i, EmbeddingUnavailableError(fatal, fatal=True))
                continue
            usable = []
            for i in idxs:
                if not texts[i] or not texts[i].strip():
                    fail(i, InvalidEmbeddingError("empty text cannot be embedded"))
                else:
                    usable.append(i)
            if not usable:
                continue
            try:
                raw = prov.embed([texts[i].strip() for i in usable], model, identity.dim)
                if len(raw) != len(usable):
                    raise InvalidEmbeddingError(
                        f"provider returned {len(raw)} embeddings for {len(usable)} inputs"
                    )
                pairs = list(zip(usable, raw))
            except EmbeddingError as exc:
                # Batch failed as a whole: retry item by item so one bad input
                # does not take down its neighbours.
                pairs = []
                if isinstance(exc, EmbeddingUnavailableError) and exc.fatal:
                    fatal = str(exc)
                for i in usable:
                    if fatal is not None:
                        fail(i, EmbeddingUnavailableError(fatal, fatal=True))
                        continue
                    try:
                        one = prov.embed([texts[i].strip()], model, identity.dim)
                        if len(one) != 1:
                            raise InvalidEmbeddingError(
                                f"provider returned {len(one)} embeddings for 1 input"
                            )
                        pairs.append((i, one[0]))
                    except EmbeddingError as inner:
                        if isinstance(inner, EmbeddingUnavailableError) and inner.fatal:
                            fatal = str(inner)
                        fail(i, inner)
            for i, vec in pairs:
                try:
                    vectors[i] = validate_vector(vec, identity.dim)
                except InvalidEmbeddingError as exc:
                    fail(i, exc)

        failures.sort(key=lambda f: f.index)
        return EmbeddingBatch(vectors=vectors, identity=identity, failures=failures)

    def embed_single(self, text: str, model: str = PROSE_MODEL) -> list[float]:
        """Embed one text; raises a typed EmbeddingError on any failure."""
        batch = self.embed_texts([text], model)
        if batch.failures:
            f = batch.failures[0]
            if f.error_type == "InvalidEmbeddingError":
                raise InvalidEmbeddingError(f.message)
            raise EmbeddingUnavailableError(f.message)
        return batch.vectors[0]  # type: ignore[return-value]

    def embed_batch(
        self,
        texts: list[str],
        model: str = PROSE_MODEL,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed all *texts* or raise EmbeddingBatchError (never returns placeholders)."""
        return self.embed_texts(texts, model, batch_size=batch_size).require_all()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if any configured provider is reachable."""
        return any(self._providers[n].reachable() for n in self.provider_order)

    def has_model(self, model: str) -> bool:
        """True if a configured provider can serve *model*."""
        return any(self._providers[n].available(model) for n in self.provider_order)

    def close(self) -> None:
        self._client.close()
