from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from config.settings import settings


@dataclass
class LLMResponse:
    text: str
    model: str


class LLMClient:
    """Minimal LLM client with Gemini primary and Ollama fallback."""

    def __init__(
        self,
        gemini_model: str = "gemini-2.5-flash",
        ollama_model: str = "qwen2.5:1.5b",
        ollama_base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.ollama_base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout
        self._http = httpx.Client(timeout=timeout)
        self._gemini = None

        if settings.gemini_api_key:
            try:
                from google import genai  # type: ignore

                self._gemini = genai.Client(api_key=settings.gemini_api_key)
            except Exception:
                self._gemini = None



    def generate(self, prompt: str) -> LLMResponse:
        if self._gemini is not None:
            import time
            for attempt in range(5):
                try:
                    resp = self._gemini.models.generate_content(
                        model=self.gemini_model,
                        contents=prompt,
                    )
                    return LLMResponse(text=resp.text or "", model=self.gemini_model)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        if attempt < 4:
                            sleep_time = (2 ** attempt) * 5
                            print(f"  [!] Gemini generation rate limited (429). Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                    print(f"  [!] Warning: Gemini {self.gemini_model} generation failed. {e}")
                    break

        try:
            print(f"  [!] Falling back to local Ollama model: {self.ollama_model}...")
            text = self._ollama_generate(prompt)
            return LLMResponse(text=text, model=self.ollama_model)
        except Exception as e:
            print(f"  [!] Warning: Ollama generation failed. Returning empty response. {e}")
            return LLMResponse(text="", model=self.ollama_model)

    def _ollama_generate(self, prompt: str) -> str:
        resp = self._http.post(
            f"{self.ollama_base_url}/api/generate",
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def close(self) -> None:
        self._http.close()
