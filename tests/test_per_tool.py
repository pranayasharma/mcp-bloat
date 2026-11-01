"""Tests for the per-tool token breakdown (server -> tool drill-down)."""

import asyncio
import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from rich.console import Console

from mcp_bloat import cli
from mcp_bloat.audit import ServerAudit, ToolInfo, _collect
from mcp_bloat.demo import run_demo
from mcp_bloat.discovery import ServerEntry
from mcp_bloat.report import (
    most_expensive_tool,
    render_terminal,
    to_json_dict,
)

_DEMO_CACHE: list[ServerAudit] | None = None


def _demo_results() -> list[ServerAudit]:
    """Run the real demo audit once and share it across tests."""
    global _DEMO_CACHE
    if _DEMO_CACHE is None:
        results, _warnings = run_demo()
        _DEMO_CACHE = results
    return _DEMO_CACHE


def _make_audit(name, tool_specs, ok=True, error=None) -> ServerAudit:
    entry = ServerEntry(
        name=name, source="test-config", client="Test", transport="stdio"
    )
    audit = ServerAudit(entry=entry, ok=ok, error=error)
    for tool_name, tokens in tool_specs:
        audit.tools.append(ToolInfo(tool_name, tokens))
        audit.tools_tokens += tokens
    return audit


def _render(results, show_tools=True) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=110)
    render_terminal(results, console=console, show_tools=show_tools)
    return buffer.getvalue()


class TestPerToolInvariant(unittest.TestCase):
    def test_demo_tool_counts_sum_to_tools_tokens(self):
        for audit in _demo_results():
            self.assertTrue(audit.ok)
            self.assertGreater(audit.tool_count, 0)
            self.assertEqual(
                sum(t.tokens for t in audit.tools),
                audit.tools_tokens,
                f"per-tool counts diverge for {audit.entry.name}",
            )

    def test_collect_accumulates_exactly_the_per_tool_counts(self):
        # Feed _collect a fake session and check the same invariant at
        # the point where the counts are produced.
        fake_tools = [
            SimpleNamespace(
                name="alpha",
                description="Alpha tool.",
                inputSchema={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="beta",
                description="Beta tool with a bigger schema.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "limit": {"type": "integer", "description": "Max results."},
                    },
                    "required": ["query"],
                },
            ),
        ]

        class FakeSession:
            async def list_tools(self, **kwargs):
                return SimpleNamespace(tools=fake_tools, nextCursor=None)

            async def list_prompts(self):
                raise RuntimeError("not implemented")

            async def list_resources(self):
                raise RuntimeError("not implemented")

        entry = ServerEntry(
            name="fake", source="test", client="Test", transport="stdio"
        )
        audit = ServerAudit(entry=entry)
        asyncio.run(_collect(FakeSession(), audit))
        self.assertEqual(audit.tool_count, 2)
        self.assertEqual(
            sum(t.tokens for t in audit.tools), audit.tools_tokens
        )
        self.assertGreater(audit.tools_tokens, 0)


class TestTerminalBreakdown(unittest.TestCase):
    def _audits(self):
        return [
            _make_audit(
                "srv",
                [("alpha_small", 50), ("gamma_fat", 500), ("beta_mid", 200)],
            )
        ]

    def test_default_shows_tool_rows_sorted_desc(self):
        text = _render(self._audits())
        self.assertIn("srv — tools by cost", text)
        self.assertLess(text.index("gamma_fat"), text.index("beta_mid"))
        self.assertLess(text.index("beta_mid"), text.index("alpha_small"))
        self.assertIn("66.7%", text)  # gamma_fat: 500 of 750 tool tokens

    def test_default_shows_most_expensive_tool_line(self):
        text = _render(self._audits())
        self.assertIn(
            "Most expensive single tool: gamma_fat (srv) — 500 tokens", text
        )

    def test_summary_mode_omits_breakdown(self):
        text = _render(self._audits(), show_tools=False)
        self.assertIn("srv", text)  # the server table is still there
        self.assertNotIn("tools by cost", text)
        self.assertNotIn("gamma_fat", text)
        self.assertNotIn("Most expensive single tool", text)

    def test_errored_server_shows_no_breakdown(self):
        audits = [
            _make_audit("broken", [("ghost_tool", 999)], ok=False, error="boom"),
            _make_audit("healthy", [("real_tool", 10)]),
        ]
        text = _render(audits)
        self.assertNotIn("broken — tools by cost", text)
        self.assertNotIn("ghost_tool", text)
        self.assertIn("healthy — tools by cost", text)
        # …and the errored server's tools don't win "most expensive".
        self.assertIn("Most expensive single tool: real_tool (healthy)", text)

    def test_zero_tool_server_shows_no_breakdown(self):
        text = _render([_make_audit("empty", [])])
        self.assertNotIn("tools by cost", text)
        self.assertNotIn("Most expensive single tool", text)


class TestJsonPerTool(unittest.TestCase):
    def test_tools_sorted_with_percent_math(self):
        audit = _make_audit(
            "srv", [("small", 200), ("big", 500), ("mid", 300)]
        )
        report = to_json_dict([audit])
        tools = report["servers"][0]["tools"]
        self.assertEqual(
            [t["name"] for t in tools], ["big", "mid", "small"]
        )
        self.assertEqual(
            [t["percent_of_server"] for t in tools], [50.0, 30.0, 20.0]
        )
        self.assertAlmostEqual(
            sum(t["percent_of_server"] for t in tools), 100.0, places=6
        )

    def test_most_expensive_tool_across_servers(self):
        audits = [
            _make_audit("one", [("a", 100), ("b", 700)]),
            _make_audit("two", [("c", 300)]),
        ]
        report = to_json_dict(audits)
        self.assertEqual(
            report["most_expensive_tool"],
            {"server": "one", "tool": "b", "tokens": 700},
        )
        top = most_expensive_tool(audits)
        self.assertIsNotNone(top)
        self.assertEqual(top[1].name, "b")

    def test_most_expensive_tool_null_without_tools(self):
        self.assertIsNone(to_json_dict([])["most_expensive_tool"])
        self.assertIsNone(
            to_json_dict([_make_audit("empty", [])])["most_expensive_tool"]
        )

    def test_most_expensive_tool_ignores_errored_servers(self):
        audits = [
            _make_audit("broken", [("ghost", 999)], ok=False, error="boom"),
            _make_audit("healthy", [("real", 10)]),
        ]
        report = to_json_dict(audits)
        self.assertEqual(
            report["most_expensive_tool"],
            {"server": "healthy", "tool": "real", "tokens": 10},
        )

    def test_demo_json_tools(self):
        report = to_json_dict(_demo_results())
        by_name = {s["name"]: s for s in report["servers"]}
        toolbox = by_name["dev-toolbox"]
        tokens = [t["tokens"] for t in toolbox["tools"]]
        self.assertEqual(tokens, sorted(tokens, reverse=True))
        self.assertEqual(len(tokens), toolbox["tool_count"])
        # Per-tool percentages of the server's tool bill add up to 100.
        self.assertAlmostEqual(
            sum(t["percent_of_server"] for t in toolbox["tools"]),
            100.0,
            delta=0.1,
        )
        biggest = max(
            t.tokens for audit in _demo_results() for t in audit.tools
        )
        self.assertEqual(report["most_expensive_tool"]["server"], "mega-api")
        self.assertEqual(report["most_expensive_tool"]["tokens"], biggest)


class TestCliSummaryFlag(unittest.TestCase):
    def test_parser_summary_flag(self):
        parser = cli._build_parser()
        self.assertFalse(parser.parse_args([]).summary)
        self.assertTrue(parser.parse_args(["--summary"]).summary)

    def _run_demo_cli(self, extra_args) -> str:
        buffer = io.StringIO()
        env = dict(os.environ, COLUMNS="200")
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(cli, "resolve_frontier", return_value=[]):
                with redirect_stdout(buffer):
                    code = cli.main(["--demo", *extra_args])
        self.assertEqual(code, 0)
        return buffer.getvalue()

    def test_cli_demo_default_shows_tools(self):
        text = self._run_demo_cli([])
        self.assertIn("dev-toolbox — tools by cost", text)
        self.assertIn("run_command", text)
        self.assertIn("Most expensive single tool", text)

    def test_cli_demo_summary_hides_tools(self):
        text = self._run_demo_cli(["--summary"])
        self.assertIn("dev-toolbox", text)
        self.assertNotIn("tools by cost", text)
        self.assertNotIn("run_command", text)
        self.assertNotIn("Most expensive single tool", text)


if __name__ == "__main__":
    unittest.main()
