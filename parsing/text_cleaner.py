"""text_cleaner.py — Normalize raw Markdown / HTML from PR bodies and issue text."""
from __future__ import annotations

import re

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    import markdownify as _md
    _MARKDOWNIFY_AVAILABLE = True
except ImportError:
    _MARKDOWNIFY_AVAILABLE = False


class TextCleaner:
    """Clean and normalize text from GitHub issue/PR bodies.

    Falls back gracefully when optional deps (bs4, markdownify) are absent.
    """

    # Collapse runs of 3+ blank lines to 2
    _MULTI_BLANK = re.compile(r"\n{3,}")
    # Strip GitHub checkbox syntax
    _CHECKBOX = re.compile(r"- \[[ xX]\] ")
    # Strip HTML-style comments <!-- ... -->
    _HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

    def clean_html(self, html: str) -> str:
        """Convert HTML to plain text, stripping all tags."""
        if not html:
            return ""
        if _MARKDOWNIFY_AVAILABLE:
            text = _md.markdownify(html, strip=["a", "img"])
        elif _BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
        else:
            # Fallback: strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html)
        return self._normalize(text)

    def clean_markdown(self, text: str) -> str:
        """Light normalization of Markdown prose (does NOT strip formatting)."""
        if not text:
            return ""
        text = self._HTML_COMMENT.sub("", text)
        text = self._CHECKBOX.sub("", text)
        return self._normalize(text)

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._MULTI_BLANK.sub("\n\n", text)
        return text.strip()
