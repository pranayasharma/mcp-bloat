"""Model context-window lookup.

Resolves a model name (``--model gpt-5``) to its context-window size so
report percentages can be computed against the window of the model you
actually use, instead of a fixed 200k default.

Data sources, in order (both free, no API key, fetched with stdlib
urllib only):

1. OpenRouter — GET https://openrouter.ai/api/v1/models
2. LiteLLM — GET https://raw.githubusercontent.com/BerriAI/litellm/main/
   model_prices_and_context_window.json

The normalized dataset is cached at ~/.cache/mcp-bloat/models.json and
reused for 24h. Every failure mode silent-degrades: fresh cache, then
stale cache, then a small built-in snapshot of major models. Lookup must
never crash or block an audit.
"""

from __future__ import annotations

import difflib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
FETCH_TIMEOUT = 10  # seconds, per source
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_RELATIVE_PATH = Path("mcp-bloat") / "models.json"

DEFAULT_CONTEXT_WINDOW = 200_000

_USER_AGENT = "mcp-bloat (+https://github.com/pranayasharma/mcp-bloat)"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    context_length: int
    source: str  # "openrouter" | "litellm" | "built-in snapshot" | "cache"


@dataclass
class Dataset:
    models: dict[str, ModelInfo]
    source: str  # where the data came from, for display
    available: bool  # False only when nothing could be fetched or cached


@dataclass
class Resolution:
    info: ModelInfo | None
    candidates: list[ModelInfo] = field(default_factory=list)
    data_available: bool = True


# ---------------------------------------------------------------------------
# Built-in snapshot fallback.
#
# Values are OpenRouter `context_length` figures pulled live from
# https://openrouter.ai/api/v1/models on 2026-10-31 (cross-checked against
# LiteLLM's model_prices_and_context_window.json the same day; where the two
# disagree, OpenRouter's total context length is used, since that is what
# this tool measures percentages against). Refresh when bumping versions.
# ---------------------------------------------------------------------------
_SNAPSHOT_ROWS: list[tuple[str, str, int]] = [
    # (id, display name, context_length)
    ("anthropic/claude-sonnet-5.5", "Anthropic: Claude Sonnet 5.5", 1_000_000),
    ("anthropic/claude-sonnet-4.5", "Anthropic: Claude Sonnet 4.5", 1_000_000),
    ("anthropic/claude-opus-5.5", "Anthropic: Claude Opus 5.5", 1_000_000),
    ("anthropic/claude-opus-4.5", "Anthropic: Claude Opus 4.5", 200_000),
    ("anthropic/claude-haiku-5.5", "Anthropic: Claude Haiku 5.5", 1_000_000),
    ("anthropic/claude-haiku-4.5", "Anthropic: Claude Haiku 4.5", 200_000),
    ("openai/gpt-5", "OpenAI: GPT-5", 400_000),
    ("openai/gpt-5.1", "OpenAI: GPT-5.1", 400_000),
    ("openai/gpt-5-mini", "OpenAI: GPT-5 Mini", 400_000),
    ("openai/gpt-5-nano", "OpenAI: GPT-5 Nano", 400_000),
    ("openai/gpt-4o", "OpenAI: GPT-4o", 128_000),
    ("openai/gpt-4o-mini", "OpenAI: GPT-4o-mini", 128_000),
    ("google/gemini-2.5-pro", "Google: Gemini 2.5 Pro", 1_048_576),
    ("google/gemini-2.5-flash", "Google: Gemini 2.5 Flash", 1_048_576),
    ("google/gemini-2.5-flash-lite", "Google: Gemini 2.5 Flash Lite", 1_048_576),
    ("meta-llama/llama-3.3-70b-instruct", "Meta: Llama 3.3 70B Instruct", 131_072),
    ("meta-llama/llama-4-maverick", "Meta: Llama 4 Maverick", 524_288),
    ("meta-llama/llama-4-scout", "Meta: Llama 4 Scout", 1_310_720),
    ("deepseek/deepseek-chat", "DeepSeek: DeepSeek V3", 163_840),
    ("deepseek/deepseek-r1-0528", "DeepSeek: R1 0528", 163_840),
    ("qwen/qwen3-coder", "Qwen: Qwen3 Coder 480B A35B", 262_144),
    ("qwen/qwen3-32b", "Qwen: Qwen3 32B", 131_072),
    ("qwen/qwen-plus", "Qwen: Qwen-Plus", 1_000_000),
]

BUILTIN_SNAPSHOT: dict[str, ModelInfo] = {
    model_id: ModelInfo(
        id=model_id,
        name=name,
        context_length=context,
        source="built-in snapshot",
    )
    for model_id, name, context in _SNAPSHOT_ROWS
}


# ---------------------------------------------------------------------------
# Fetching / parsing
# ---------------------------------------------------------------------------


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_openrouter(payload: object) -> dict[str, ModelInfo]:
    models: dict[str, ModelInfo] = {}
    if not isinstance(payload, dict):
        return models
    data = payload.get("data")
    if not isinstance(data, list):
        return models
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        context = item.get("context_length")
        if not isinstance(model_id, str) or not model_id:
            continue
        if not isinstance(context, (int, float)) or context <= 0:
            continue
        name = item.get("name")
        models[model_id] = ModelInfo(
            id=model_id,
            name=name if isinstance(name, str) and name else model_id,
            context_length=int(context),
            source="openrouter",
        )
    return models


def parse_litellm(payload: object) -> dict[str, ModelInfo]:
    models: dict[str, ModelInfo] = {}
    if not isinstance(payload, dict):
        return models
    for model_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        context = entry.get("max_input_tokens")
        if not isinstance(context, (int, float)) or context <= 0:
            continue
        # LiteLLM has no display name; the id doubles as the name.
        models[model_id] = ModelInfo(
            id=model_id,
            name=model_id,
            context_length=int(context),
            source="litellm",
        )
    return models


# ---------------------------------------------------------------------------
# Cache (~/.cache/mcp-bloat/models.json)
# ---------------------------------------------------------------------------


def cache_path(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir) / CACHE_RELATIVE_PATH
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / CACHE_RELATIVE_PATH


def _read_cache(cache_dir: Path | None) -> tuple[dict[str, ModelInfo], str, float] | None:
    try:
        raw = json.loads(cache_path(cache_dir).read_text(encoding="utf-8"))
        fetched_at = float(raw["fetched_at"])
        source = str(raw.get("source", "cache"))
        models = {
            str(item["id"]): ModelInfo(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                context_length=int(item["context_length"]),
                source="cache",
            )
            for item in raw["models"]
        }
        if not models:
            return None
        return models, source, fetched_at
    except Exception:
        return None


def _write_cache(
    cache_dir: Path | None, models: Mapping[str, ModelInfo], source: str
) -> None:
    try:
        path = cache_path(cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "source": source,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "context_length": m.context_length,
                }
                for m in models.values()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass  # caching is best-effort


def load_dataset(cache_dir: Path | None = None) -> Dataset:
    """Best available model dataset: live fetch, then cache, else empty."""
    cached = _read_cache(cache_dir)
    if cached is not None:
        models, source, fetched_at = cached
        if time.time() - fetched_at < CACHE_TTL_SECONDS:
            return Dataset(models=models, source=f"{source} (cached)", available=True)

    for label, url, parser in (
        ("openrouter", OPENROUTER_URL, parse_openrouter),
        ("litellm", LITELLM_URL, parse_litellm),
    ):
        try:
            models = parser(_fetch_json(url))
        except Exception:
            continue
        if models:
            _write_cache(cache_dir, models, label)
            return Dataset(models=models, source=label, available=True)

    if cached is not None:
        models, source, _ = cached
        return Dataset(models=models, source=f"{source} (stale cache)", available=True)
    return Dataset(models={}, source="none", available=False)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Normalize for comparison: lowercase, unify separators.

    Makes ``claude-sonnet-4-5`` match ``anthropic/claude-sonnet-4.5``.
    """
    return text.strip().lower().replace(".", "-").replace("_", "-")


def _suffix(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _dedupe_candidates(candidates: list[ModelInfo]) -> list[ModelInfo]:
    """Drop ``:variant`` duplicates and repeated ids; keep it short."""
    plain = [c for c in candidates if ":" not in c.id]
    pool = plain if plain else candidates
    seen: set[str] = set()
    unique: list[ModelInfo] = []
    for candidate in sorted(pool, key=lambda c: c.id.lower()):
        key = candidate.id.lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:8]


def find_model(
    query: str, models: Mapping[str, ModelInfo]
) -> tuple[ModelInfo | None, list[ModelInfo]]:
    """Match a user-typed model name against a dataset.

    Returns (match, []) on a confident hit, (None, candidates) when the
    name is ambiguous or close-but-not-found, (None, []) when nothing is
    remotely close.
    """
    q = _norm(query)
    if not q or not models:
        return None, []
    by_norm_id = {_norm(m.id): m for m in models.values()}

    # 1. Exact id.
    if q in by_norm_id:
        return by_norm_id[q], []

    # 2. Exact suffix after the provider "/".
    suffix_hits = [
        m for m in models.values() if _norm(_suffix(m.id)) == q
    ]
    if suffix_hits:
        if len({m.context_length for m in suffix_hits}) == 1:
            return min(suffix_hits, key=lambda m: len(m.id)), []
        return None, _dedupe_candidates(suffix_hits)

    # 3. Substring on id or display name.
    substring_hits = [
        m
        for m in models.values()
        if q in _norm(m.id) or q in _norm(m.name)
    ]
    if substring_hits:
        deduped = _dedupe_candidates(substring_hits)
        if len(deduped) == 1:
            return deduped[0], []
        return None, deduped

    # 4. Nothing close by substring — suggest nearest names.
    names = sorted({_norm(_suffix(m.id)) for m in models.values()})
    close = difflib.get_close_matches(q, names, n=6, cutoff=0.55)
    suggestions = [
        by_norm_id[norm_id]
        for norm_id, m in by_norm_id.items()
        if _norm(_suffix(m.id)) in close
    ]
    return None, _dedupe_candidates(suggestions)


def resolve_model(query: str, cache_dir: Path | None = None) -> Resolution:
    """Resolve a user-typed model name to a context window.

    Never raises. Live/cached data is searched first; the built-in
    snapshot fills any gaps (and is the only source when offline).
    """
    try:
        dataset = load_dataset(cache_dir)
    except Exception:
        dataset = Dataset(models={}, source="none", available=False)

    merged: dict[str, ModelInfo] = dict(BUILTIN_SNAPSHOT)
    merged.update(dataset.models)  # fetched data wins over the snapshot

    info, candidates = find_model(query, merged)
    if info is not None:
        return Resolution(info=info, data_available=True)
    if not dataset.available:
        # Offline with no snapshot hit: we cannot tell a typo from a real
        # model we simply don't know, so the caller falls back to the
        # default window instead of erroring out.
        return Resolution(info=None, candidates=[], data_available=False)
    return Resolution(info=None, candidates=candidates, data_available=True)
