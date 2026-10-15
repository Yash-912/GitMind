from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from config.settings import settings

log = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Every configured generation provider failed or returned no text."""

    def __init__(self, causes: list[str]) -> None:
        self.causes = causes
        super().__init__("No LLM provider produced a response: " + "; ".join(causes))


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str = ""
    fallback_used: bool = False


class LLMClient:
    """LLM client with Gemini primary and Ollama fallback.

    A total outage raises LLMUnavailableError; it is never reported as an
    empty successful response. `deadline_s` bounds the whole call, including
    rate-limit back-off.
    """

    def __init__(
        self,
        gemini_model: str = "gemini-2.5-flash",
        ollama_model: str = "qwen2.5:1.5b",
        ollama_base_url: str | None = None,
        timeout: float = 120.0,
        deadline_s: float | None = 180.0,
        max_attempts: int = 5,
    ) -> None:
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.ollama_base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout
        self.deadline_s = deadline_s
        self.max_attempts = max_attempts
        self._http = httpx.Client(timeout=timeout)
        self._gemini = None
        self._sleep = time.sleep
        self._clock = time.monotonic

        if settings.gemini_api_key:
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore

                self._gemini = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                )
            except Exception as exc:  # pragma: no cover - depends on SDK version
                log.warning("Gemini client unavailable: %s", exc)
                self._gemini = None

    @property
    def providers(self) -> list[str]:
        """Provider names that are configured (not necessarily reachable)."""
        names = []
        if self._gemini is not None:
            names.append("gemini")
        names.append("ollama")
        return names

    def generate(self, prompt: str, *, deadline_s: float | None = None) -> LLMResponse:
        budget = self.deadline_s if deadline_s is None else deadline_s
        start = self._clock()

        def remaining() -> float | None:
            return None if budget is None else budget - (self._clock() - start)

        causes: list[str] = []

        if self._gemini is not None:
            for attempt in range(self.max_attempts):
                rem = remaining()
                if rem is not None and rem <= 0:
                    causes.append("gemini: deadline exceeded")
                    break
                try:
                    resp = self._gemini.models.generate_content(
                        model=self.gemini_model, contents=prompt
                    )
                    text = (resp.text or "").strip()
                    if text:
                        return LLMResponse(
                            text=text, model=self.gemini_model, provider="gemini"
                        )
                    causes.append("gemini: empty response")
                    break
                except Exception as exc:
                    err = str(exc)
                    rate_limited = (
                        "429" in err
                        or "RESOURCE_EXHAUSTED" in err
                        or "quota" in err.lower()
                    )
                    if rate_limited and attempt < self.max_attempts - 1:
                        wait = (2 ** attempt) * 5
                        rem = remaining()
                        if rem is not None and wait >= rem:
                            causes.append("gemini: rate limited, no time left to retry")
                            break
                        log.warning("Gemini rate limited; retrying in %ss", wait)
                        self._sleep(wait)
                        continue
                    causes.append(f"gemini: {err[:200]}")
                    break

        rem = remaining()
        if rem is not None and rem <= 0:
            causes.append("ollama: skipped, deadline exceeded")
            raise LLMUnavailableError(causes)
        try:
            text = self._ollama_generate(prompt).strip()
            if text:
                return LLMResponse(
                    text=text,
                    model=self.ollama_model,
                    provider="ollama",
                    fallback_used=self._gemini is not None,
                )
            causes.append("ollama: empty response")
        except Exception as exc:
            causes.append(f"ollama: {str(exc)[:200]}")
        raise LLMUnavailableError(causes)

    def _ollama_generate(self, prompt: str) -> str:
        resp = self._http.post(
            f"{self.ollama_base_url}/api/generate",
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def close(self) -> None:
        self._http.close()
