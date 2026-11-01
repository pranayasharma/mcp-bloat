"""Token counting for tool schemas.

Prefers tiktoken (cl100k_base) when installed; falls back to a chars/4
estimate and says so, so the number on screen is never silently wrong
about how it was produced.
"""

from __future__ import annotations

import math

_encoder = None
_method = "estimate (chars/4 — pip install 'mcp-bloat[accurate]' for exact counts)"

try:  # optional dependency
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")
    _method = "tiktoken cl100k_base"
except Exception:  # pragma: no cover - tiktoken is optional
    _encoder = None


def counting_method() -> str:
    return _method


def count_tokens(text: str) -> int:
    if _encoder is not None:
        return len(_encoder.encode(text))
    if not text:
        return 0
    return math.ceil(len(text) / 4)
