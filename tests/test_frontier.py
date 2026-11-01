"""Tests for the frontier-model comparison in the default summary."""

import io
import unittest
from unittest import mock

from rich.console import Console

from mcp_bloat import models
from mcp_bloat.audit import ServerAudit, ToolInfo
from mcp_bloat.discovery import ServerEntry
from mcp_bloat.models import (
    BUILTIN_SNAPSHOT,
    Dataset,
    ModelInfo,
    frontier_from_models,
    resolve_frontier,
    short_display_name,
)
from mcp_bloat.report import render_terminal, to_json_dict

FIXTURE_PAYLOAD = {
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
            "id": "anthropic/claude-sonnet-5.5",
            "name": "Anthropic: Claude Sonnet 5.5",
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
        {
            "id": "google/gemini-2.5-pro",
            "name": "Google: Gemini 2.5 Pro",
            "context_length": 1_048_576,
        },
        {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek: DeepSeek V3",
            "context_length": 163_840,
        },
    ]
}


def _fixture() -> dict[str, ModelInfo]:
    return models.parse_openrouter(FIXTURE_PAYLOAD)


def _frontier_models() -> list[ModelInfo]:
    return frontier_from_models(_fixture())


class TestFrontierResolution(unittest.TestCase):
    def test_all_families_resolve_in_order(self):
        resolved = _frontier_models()
        self.assertEqual(
            [m.id for m in resolved],
            [
                "anthropic/claude-sonnet-5.5",
                "openai/gpt-5",
                "google/gemini-2.5-pro",
                "deepseek/deepseek-chat",
            ],
        )

    def test_ambiguous_family_prefers_current_generation(self):
        # "claude-sonnet" matches Sonnet 4.5 and 5.5; the newer
        # generation wins, and the ":batch" variant is never picked.
        resolved = _frontier_models()
        sonnet = resolved[0]
        self.assertEqual(sonnet.id, "anthropic/claude-sonnet-5.5")
        self.assertEqual(sonnet.context_length, 1_000_000)

    def test_missing_families_are_skipped(self):
        dataset = {
            k: v
            for k, v in _fixture().items()
            if not k.startswith(("google/", "deepseek/"))
        }
        resolved = frontier_from_models(dataset)
        self.assertEqual(
            [m.id for m in resolved],
            ["anthropic/claude-sonnet-5.5", "openai/gpt-5"],
        )

    def test_empty_dataset_resolves_nothing(self):
        self.assertEqual(frontier_from_models({}), [])

    def test_duplicate_resolution_appears_once(self):
        with mock.patch.object(
            models, "FRONTIER_QUERIES", ["gpt-5", "gpt-5", "gpt-5"]
        ):
            resolved = frontier_from_models(_fixture())
        self.assertEqual([m.id for m in resolved], ["openai/gpt-5"])

    def test_snapshot_alone_resolves_all_families(self):
        # The offline path: built-in snapshot only, no live data.
        resolved = frontier_from_models(BUILTIN_SNAPSHOT)
        self.assertEqual(len(resolved), 4)
        windows = {m.id: m.context_length for m in resolved}
        self.assertEqual(windows["openai/gpt-5"], 400_000)

    def test_resolve_frontier_offline_uses_snapshot(self):
        offline = Dataset(models={}, source="none", available=False)
        with mock.patch.object(models, "load_dataset", return_value=offline):
            resolved = resolve_frontier()
        self.assertEqual(len(resolved), 4)
        self.assertTrue(all(m.source == "built-in snapshot" for m in resolved))

    def test_short_display_name(self):
        sonnet = ModelInfo(
            id="anthropic/claude-sonnet-5.5",
            name="Anthropic: Claude Sonnet 5.5",
            context_length=1_000_000,
            source="openrouter",
        )
        self.assertEqual(short_display_name(sonnet), "Claude Sonnet 5.5")
        # LiteLLM style: name == id, so the id suffix is the label.
        litellm = ModelInfo(
            id="gpt-5", name="gpt-5", context_length=400_000, source="litellm"
        )
        self.assertEqual(short_display_name(litellm), "gpt-5")


class TestFrontierReport(unittest.TestCase):
    TOTAL = 27_762

    def _audit(self, tokens: int) -> ServerAudit:
        entry = ServerEntry(
            name="demo", source="test", client="Test", transport="stdio"
        )
        audit = ServerAudit(entry=entry, ok=True, tools=[ToolInfo("t", tokens)])
        audit.tools_tokens = tokens
        return audit

    def _render(self, frontier) -> str:
        buffer = io.StringIO()
        console = Console(file=buffer, width=110)
        render_terminal(
            [self._audit(self.TOTAL)], console=console, frontier=frontier
        )
        return buffer.getvalue()

    def test_terminal_shows_frontier_block_in_default_mode(self):
        text = self._render(_frontier_models())
        self.assertIn(f"Of your {self.TOTAL:,} tokens, that's:", text)
        self.assertIn("2.8% of Claude Sonnet 5.5's 1,000,000 window", text)
        self.assertIn("6.9% of GPT-5's 400,000 window", text)
        self.assertIn("2.6% of Gemini 2.5 Pro's 1,048,576 window", text)
        self.assertIn("16.9% of DeepSeek V3's 163,840 window", text)

    def test_terminal_omits_frontier_block_in_single_window_mode(self):
        # --window/--model mode passes frontier=None.
        text = self._render(None)
        self.assertNotIn("Of your", text)
        self.assertNotIn("Claude Sonnet 5.5's", text)

    def test_terminal_omits_frontier_block_when_unresolved(self):
        text = self._render([])
        self.assertNotIn("Of your", text)

    def test_json_frontier_entries_and_math(self):
        report = to_json_dict(
            [self._audit(self.TOTAL)], frontier=_frontier_models()
        )
        frontier = report["frontier"]
        self.assertEqual(len(frontier), 4)
        by_id = {entry["model"]: entry for entry in frontier}
        gpt = by_id["openai/gpt-5"]
        self.assertEqual(gpt["name"], "OpenAI: GPT-5")
        self.assertEqual(gpt["context_window"], 400_000)
        self.assertAlmostEqual(gpt["percent_of_context"], 6.94, places=2)
        sonnet = by_id["anthropic/claude-sonnet-5.5"]
        self.assertAlmostEqual(sonnet["percent_of_context"], 2.78, places=2)

    def test_json_frontier_absent_in_single_window_mode(self):
        report = to_json_dict([self._audit(self.TOTAL)])
        self.assertNotIn("frontier", report)

    def test_json_frontier_empty_list_when_unresolved(self):
        report = to_json_dict([self._audit(self.TOTAL)], frontier=[])
        self.assertEqual(report["frontier"], [])


if __name__ == "__main__":
    unittest.main()
