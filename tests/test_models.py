"""Tests for model context-window lookup, caching, and window-aware reports."""

import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from mcp_bloat import models
from mcp_bloat.audit import ServerAudit, ToolInfo
from mcp_bloat.discovery import ServerEntry
from mcp_bloat.models import ModelInfo, find_model, load_dataset, parse_litellm
from mcp_bloat.report import percent_of_context, to_json_dict

OPENROUTER_FIXTURE = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "context_length": 1_000_000,
        },
        {
            "id": "anthropic/claude-sonnet-4.5:batch",
            "name": "Anthropic: Claude Sonnet 4.5 (batch)",
            "context_length": 1_000_000,
        },
        {
            "id": "openai/gpt-5",
            "name": "OpenAI: GPT-5",
            "context_length": 400_000,
        },
        {
            "id": "openai/gpt-5-mini",
            "name": "OpenAI: GPT-5 Mini",
            "context_length": 400_000,
        },
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128_000},
        # Broken entries must be skipped, not crash the parse.
        {"id": "broken/no-context"},
        {"name": "no id", "context_length": 1000},
    ]
}

LITELLM_FIXTURE = {
    "gpt-4o": {"max_input_tokens": 128_000},
    "some/embedding-model": {"mode": "embedding"},  # no token count: skipped
}


def _dataset() -> dict[str, ModelInfo]:
    return models.parse_openrouter(OPENROUTER_FIXTURE)


class TestParsing(unittest.TestCase):
    def test_openrouter_fixture(self):
        parsed = _dataset()
        self.assertEqual(parsed["openai/gpt-5"].context_length, 400_000)
        self.assertEqual(parsed["openai/gpt-5"].source, "openrouter")
        self.assertNotIn("broken/no-context", parsed)

    def test_litellm_fixture(self):
        parsed = parse_litellm(LITELLM_FIXTURE)
        self.assertEqual(list(parsed), ["gpt-4o"])
        self.assertEqual(parsed["gpt-4o"].context_length, 128_000)
        self.assertEqual(parsed["gpt-4o"].source, "litellm")


class TestFindModel(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()

    def test_exact_id(self):
        info, candidates = find_model("openai/gpt-5", self.dataset)
        self.assertIsNotNone(info)
        self.assertEqual(info.context_length, 400_000)
        self.assertEqual(candidates, [])

    def test_suffix_match(self):
        info, _ = find_model("gpt-4o", self.dataset)
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "openai/gpt-4o")

    def test_suffix_match_with_dash_separators(self):
        # Users type "claude-sonnet-4-5"; the id uses a dot.
        info, _ = find_model("claude-sonnet-4-5", self.dataset)
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "anthropic/claude-sonnet-4.5")
        self.assertEqual(info.context_length, 1_000_000)

    def test_case_insensitive(self):
        info, _ = find_model("GPT-5", self.dataset)
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "openai/gpt-5")

    def test_substring_unique(self):
        info, _ = find_model("sonnet-4", self.dataset)
        self.assertIsNotNone(info)
        self.assertEqual(info.id, "anthropic/claude-sonnet-4.5")

    def test_substring_ambiguous_returns_candidates(self):
        info, candidates = find_model("gpt", self.dataset)
        self.assertIsNone(info)
        ids = {c.id for c in candidates}
        self.assertIn("openai/gpt-5", ids)
        self.assertIn("openai/gpt-5-mini", ids)
        self.assertIn("openai/gpt-4o", ids)

    def test_no_match_suggests_close_names(self):
        info, candidates = find_model("gpt4o", self.dataset)
        self.assertIsNone(info)
        self.assertIn("openai/gpt-4o", {c.id for c in candidates})

    def test_garbage_matches_nothing(self):
        info, candidates = find_model("zzz-not-a-model", self.dataset)
        self.assertIsNone(info)
        self.assertEqual(candidates, [])


class TestCache(unittest.TestCase):
    def test_fresh_cache_is_used_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            models._write_cache(cache_dir, _dataset(), "openrouter")
            with mock.patch.object(
                models,
                "_fetch_json",
                side_effect=AssertionError("network must not be touched"),
            ):
                dataset = load_dataset(cache_dir)
            self.assertTrue(dataset.available)
            self.assertIn("openai/gpt-5", dataset.models)
            self.assertIn("cached", dataset.source)

    def test_cache_round_trip_through_disk_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            models._write_cache(cache_dir, _dataset(), "openrouter")
            raw = json.loads(
                models.cache_path(cache_dir).read_text(encoding="utf-8")
            )
            self.assertIn("fetched_at", raw)
            cached = models._read_cache(cache_dir)
            self.assertIsNotNone(cached)
            cached_models, source, _ = cached
            self.assertEqual(source, "openrouter")
            self.assertEqual(
                cached_models["openai/gpt-4o"].context_length, 128_000
            )
            self.assertEqual(cached_models["openai/gpt-4o"].source, "cache")

    def test_stale_cache_used_when_network_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            models._write_cache(cache_dir, _dataset(), "openrouter")
            # Age the cache beyond the TTL.
            path = models.cache_path(cache_dir)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["fetched_at"] = time.time() - models.CACHE_TTL_SECONDS - 60
            path.write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch.object(
                models,
                "_fetch_json",
                side_effect=urllib.error.URLError("offline"),
            ):
                dataset = load_dataset(cache_dir)
            self.assertTrue(dataset.available)
            self.assertIn("stale cache", dataset.source)
            self.assertIn("openai/gpt-5", dataset.models)

    def test_offline_with_no_cache_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                models,
                "_fetch_json",
                side_effect=urllib.error.URLError("offline"),
            ):
                dataset = load_dataset(Path(tmp))
            self.assertFalse(dataset.available)
            self.assertEqual(dataset.models, {})

    def test_live_fetch_writes_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with mock.patch.object(
                models, "_fetch_json", return_value=OPENROUTER_FIXTURE
            ) as fetch:
                dataset = load_dataset(cache_dir)
            self.assertTrue(dataset.available)
            self.assertEqual(dataset.source, "openrouter")
            fetch.assert_called_once_with(models.OPENROUTER_URL)
            self.assertTrue(models.cache_path(cache_dir).exists())


class TestReportWindow(unittest.TestCase):
    def _audit(self, tokens: int) -> ServerAudit:
        entry = ServerEntry(
            name="demo", source="test", client="Test", transport="stdio"
        )
        audit = ServerAudit(entry=entry, ok=True, tools=[ToolInfo("t", tokens)])
        audit.tools_tokens = tokens
        return audit

    def test_percent_math_with_128k_window(self):
        self.assertAlmostEqual(percent_of_context(27_762, 128_000), 21.7, places=1)

    def test_percent_math_default_window(self):
        self.assertAlmostEqual(percent_of_context(27_762), 13.9, places=1)

    def test_json_includes_model_and_window(self):
        model = ModelInfo(
            id="openai/gpt-4o",
            name="OpenAI: GPT-4o",
            context_length=128_000,
            source="openrouter",
        )
        report = to_json_dict(
            [self._audit(27_762)], context_window=128_000, model=model
        )
        self.assertEqual(report["model"], "openai/gpt-4o")
        self.assertEqual(report["context_window"], 128_000)
        self.assertEqual(report["total_tokens"], 27_762)
        self.assertAlmostEqual(report["percent_of_context"], 21.69, places=2)
        self.assertAlmostEqual(
            report["servers"][0]["percent_of_context"], 21.69, places=2
        )

    def test_json_model_defaults_to_null(self):
        report = to_json_dict([self._audit(100)])
        self.assertIsNone(report["model"])
        self.assertEqual(report["context_window"], 200_000)


if __name__ == "__main__":
    unittest.main()
