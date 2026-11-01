"""Tests for --tool NAME: single-tool lookup across the audit results."""

import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mcp_bloat import cli
from mcp_bloat.audit import ServerAudit, ToolInfo, _collect
from mcp_bloat.discovery import ServerEntry
from mcp_bloat.report import find_tool_matches, tool_to_json_dict


def _make_audit(name, tool_specs, source="test-config") -> ServerAudit:
    """tool_specs: (tool name, tokens, description_tokens, schema_tokens)."""
    entry = ServerEntry(
        name=name, source=source, client="Test", transport="stdio"
    )
    audit = ServerAudit(entry=entry, ok=True)
    for tool_name, tokens, desc, schema in tool_specs:
        audit.tools.append(ToolInfo(tool_name, tokens, desc, schema))
        audit.tools_tokens += tokens
    return audit


class TestFindToolMatches(unittest.TestCase):
    def setUp(self):
        self.results = [
            _make_audit(
                "payments",
                [("createPaymentTransaction", 315, 60, 240), ("refund", 90, 30, 55)],
                source="cursor-config",
            ),
            _make_audit(
                "shop",
                [("createOrder", 288, 55, 220), ("refund", 95, 28, 60)],
                source="claude-config",
            ),
        ]

    def test_exact_match_with_focused_fields(self):
        matches, suggestions = find_tool_matches(
            self.results, "createPaymentTransaction"
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(suggestions, [])
        audit, info = matches[0]
        self.assertEqual(audit.entry.name, "payments")
        report = tool_to_json_dict(self.results, "createPaymentTransaction")
        self.assertEqual(report["tool"], "createPaymentTransaction")
        (match,) = report["matches"]
        self.assertEqual(match["name"], "createPaymentTransaction")
        self.assertEqual(match["server"], "payments")
        self.assertEqual(match["source"], "cursor-config")
        self.assertEqual(match["tokens"], 315)
        self.assertEqual(match["description_tokens"], 60)
        self.assertEqual(match["schema_tokens"], 240)
        # 315 of the payments server's 405 tool tokens.
        self.assertAlmostEqual(match["percent_of_server"], 77.78, places=2)
        # 315 of the default 200,000-token window.
        self.assertAlmostEqual(match["percent_of_context"], 0.16, places=2)

    def test_exact_match_is_case_insensitive(self):
        matches, _ = find_tool_matches(self.results, "CREATEPAYMENTTRANSACTION")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1].name, "createPaymentTransaction")

    def test_exact_match_wins_over_substring(self):
        results = [
            _make_audit(
                "srv",
                [("test", 10, 4, 5), ("run_tests", 208, 39, 166)],
            )
        ]
        matches, _ = find_tool_matches(results, "test")
        self.assertEqual([info.name for _a, info in matches], ["test"])

    def test_substring_fallback(self):
        matches, _ = find_tool_matches(self.results, "payment")
        self.assertEqual(
            [info.name for _a, info in matches], ["createPaymentTransaction"]
        )

    def test_same_tool_on_multiple_servers_all_shown(self):
        matches, _ = find_tool_matches(self.results, "refund")
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {audit.entry.name for audit, _i in matches}, {"payments", "shop"}
        )
        report = tool_to_json_dict(self.results, "refund")
        self.assertEqual(len(report["matches"]), 2)
        self.assertEqual(
            {m["server"] for m in report["matches"]}, {"payments", "shop"}
        )

    def test_not_found_returns_suggestions(self):
        matches, suggestions = find_tool_matches(self.results, "refundd")
        self.assertEqual(matches, [])
        self.assertIn("refund", suggestions)
        self.assertLessEqual(len(suggestions), 5)

    def test_not_found_without_any_tools(self):
        matches, suggestions = find_tool_matches([], "anything")
        self.assertEqual(matches, [])
        self.assertEqual(suggestions, [])

    def test_errored_servers_are_skipped(self):
        audit = _make_audit("broken", [("ghost", 10, 4, 5)])
        audit.ok = False
        audit.error = "boom"
        matches, _ = find_tool_matches([audit], "ghost")
        self.assertEqual(matches, [])

    def test_percent_of_context_respects_window(self):
        report = tool_to_json_dict(
            self.results, "createOrder", context_window=1000
        )
        (match,) = report["matches"]
        self.assertAlmostEqual(match["percent_of_context"], 28.8, places=2)


class TestSplitCounting(unittest.TestCase):
    def test_collect_records_description_and_schema_tokens(self):
        fake_tool = SimpleNamespace(
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
        )

        class FakeSession:
            async def list_tools(self, **kwargs):
                return SimpleNamespace(tools=[fake_tool], nextCursor=None)

            async def list_prompts(self):
                raise RuntimeError("not implemented")

            async def list_resources(self):
                raise RuntimeError("not implemented")

        entry = ServerEntry(
            name="fake", source="test", client="Test", transport="stdio"
        )
        audit = ServerAudit(entry=entry)
        asyncio.run(_collect(FakeSession(), audit))
        (info,) = audit.tools
        self.assertGreater(info.description_tokens, 0)
        self.assertGreater(info.schema_tokens, 0)
        # The two parts plus the payload's JSON structure make the total:
        # the split never exceeds it and stays within a few tokens.
        split = info.description_tokens + info.schema_tokens
        self.assertLessEqual(split, info.tokens)
        self.assertLess(info.tokens - split, 10)


class TestCliToolMode(unittest.TestCase):
    def _run_cli(self, argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        env = dict(os.environ, COLUMNS="200")
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_demo_exact_match_terminal(self):
        code, out, _err = self._run_cli(["--demo", "--tool", "run_tests"])
        self.assertEqual(code, 0)
        self.assertIn("Tool lookup: run_tests", out)
        self.assertIn("dev-toolbox", out)
        self.assertIn("208", out)
        self.assertIn("description ≈", out)
        self.assertIn("schema ≈", out)
        self.assertIn("1 match for 'run_tests'", out)

    def test_demo_summary_flag_ignored_in_tool_mode(self):
        code, out, _err = self._run_cli(
            ["--demo", "--tool", "run_tests", "--summary"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Tool lookup: run_tests", out)

    def test_demo_json_shape(self):
        code, out, _err = self._run_cli(
            ["--demo", "--tool", "run_tests", "--json"]
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["tool"], "run_tests")
        (match,) = report["matches"]
        self.assertEqual(
            set(match),
            {
                "name",
                "server",
                "client",
                "source",
                "tokens",
                "percent_of_server",
                "percent_of_context",
                "description_tokens",
                "schema_tokens",
            },
        )
        self.assertEqual(match["server"], "dev-toolbox")
        self.assertEqual(match["tokens"], 208)
        self.assertGreater(match["description_tokens"], 0)
        self.assertGreater(match["schema_tokens"], 0)

    def test_demo_not_found_exit_2_with_suggestions(self):
        code, out, _err = self._run_cli(["--demo", "--tool", "run_testz"])
        self.assertEqual(code, 2)
        self.assertIn("not found", out)
        self.assertIn("run_tests", out)  # the suggested existing name

    def test_demo_not_found_json_keeps_stdout_valid(self):
        code, out, err = self._run_cli(
            ["--demo", "--tool", "run_testz", "--json"]
        )
        self.assertEqual(code, 2)
        report = json.loads(out)
        self.assertEqual(report, {"tool": "run_testz", "matches": []})
        self.assertIn("not found", err)

    def test_config_file_tool_lookup(self):
        # Build a config pointing at the real demo server script and
        # look a tool up through --config.
        from mcp_bloat.demo import build_demo_server_source

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "server.py"
            script.write_text(build_demo_server_source(), encoding="utf-8")
            config = Path(tmp) / "mcp.json"
            config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "toolbox": {
                                "command": sys.executable,
                                "args": [str(script), "medium"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            code, out, _err = self._run_cli(
                ["--config", str(config), "--tool", "grep", "--json"]
            )
        self.assertEqual(code, 0)
        report = json.loads(out)
        (match,) = report["matches"]
        self.assertEqual(match["name"], "grep")
        self.assertEqual(match["server"], "toolbox")


_HTTP_SERVER_SOURCE = '''
import sys

import uvicorn

try:  # mcp SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("url-tool-test")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@server.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"hello {name}"


class RequireAuth:
    """ASGI wrapper: 401 unless Authorization: Bearer test-token."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.lower(): v for k, v in scope.get("headers") or []}
            if headers.get(b"authorization") != b"Bearer test-token":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [[b"content-type", b"text/plain; charset=utf-8"]],
                    }
                )
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


if __name__ == "__main__":
    port = int(sys.argv[1])
    app = server.streamable_http_app()
    uvicorn.run(RequireAuth(app), host="127.0.0.1", port=port, log_level="error")
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestUrlToolLookup(unittest.TestCase):
    """--tool combined with --url against a local auth-gated HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="mcp-bloat-tool-url-")
        script = Path(cls._tmp.name) / "http_server.py"
        script.write_text(_HTTP_SERVER_SOURCE, encoding="utf-8")
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.proc = subprocess.Popen(
            [sys.executable, str(script), str(cls.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("HTTP test server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=1):
                    return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("HTTP test server did not start in time")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls._tmp.cleanup()

    def test_tool_lookup_over_url(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(
                [
                    "--url",
                    self.url,
                    "--header",
                    "Authorization: Bearer test-token",
                    "--tool",
                    "ADD",  # case-insensitive on purpose
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["tool"], "ADD")
        (match,) = report["matches"]
        self.assertEqual(match["name"], "add")
        self.assertEqual(match["server"], "127.0.0.1")
        self.assertGreater(match["tokens"], 0)
        self.assertGreater(match["description_tokens"], 0)
        self.assertGreater(match["schema_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
