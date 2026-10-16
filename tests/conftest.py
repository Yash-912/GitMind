"""Shared test configuration.

The embedding and generation clients build a live Gemini client from
settings.gemini_api_key when none is injected. Tests must not depend on the
developer's .env key or on Gemini quota, so the key is cleared for every test.
Tests that need a key set it explicitly with monkeypatch.
"""
from __future__ import annotations

import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def _isolate_gemini_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
