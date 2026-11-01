"""Tests for Gemini CLI / Codex CLI discovery and the --url audit mode."""

import io
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mcp_bloat.cli import main
from mcp_bloat.discovery import discover, parse_toml_config_file

try:  # Python 3.11+, or 3.10 with tomli
    import tomllib  # noqa: F401
except ImportError:
    try:
        import tomli as tomllib  # noqa: F401
    except ImportError:
        tomllib = None


def _write(tmp: str, name: str, content: str) -> Path:
    path = Path(tmp) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestGeminiDiscovery(unittest.TestCase):
    def test_gemini_home_and_project_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            cwd = Path(tmp) / "proj"
            _write(
                tmp,
                "home/.gemini/settings.json",
                json.dumps(
                    {
                        "mcpServers": {
                            "local": {
                                "command": "python",
                                "args": ["server.py"],
                                "env": {"API_KEY": "x"},
                            },
                            "hosted": {"url": "https://example.com/mcp"},
                            "hosted-alt": {
                                "httpUrl": "https://alt.example.com/mcp",
                                "headers": {"Authorization": "Bearer t"},
                            },
                        }
                    }
                ),
            )
            _write(
                tmp,
                "proj/.gemini/settings.json",
                json.dumps(
                    {"mcpServers": {"proj": {"command": "node", "args": []}}}
                ),
            )
            result = discover(home=home, cwd=cwd)
            by_name = {s.name: s for s in result.servers}
            self.assertEqual(
                sorted(by_name), ["hosted", "hosted-alt", "local", "proj"]
            )
            for entry in by_name.values():
                self.assertEqual(entry.client, "Gemini CLI")
            self.assertEqual(by_name["local"].transport, "stdio")
            self.assertEqual(by_name["local"].env, {"API_KEY": "x"})
            self.assertEqual(by_name["hosted"].transport, "http")
            self.assertEqual(by_name["hosted"].url, "https://example.com/mcp")
            # Gemini's httpUrl spelling is accepted as an HTTP server too.
            self.assertEqual(by_name["hosted-alt"].transport, "http")
            self.assertEqual(
                by_name["hosted-alt"].url, "https://alt.example.com/mcp"
            )
            self.assertEqual(
                by_name["hosted-alt"].headers, {"Authorization": "Bearer t"}
            )


@unittest.skipIf(tomllib is None, "no TOML parser available")
class TestCodexDiscovery(unittest.TestCase):
    CONFIG = """
[mcp_servers.local-tools]
command = "python"
args = ["server.py", "--verbose"]
env = { API_KEY = "x", DEBUG = "1" }

[mcp_servers.remote-api]
url = "https://example.com/mcp"
"""

    def test_codex_config_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            cwd = Path(tmp) / "proj"
            cwd.mkdir(parents=True)
            _write(tmp, "home/.codex/config.toml", self.CONFIG)
            result = discover(home=home, cwd=cwd)
            by_name = {s.name: s for s in result.servers}
            self.assertEqual(sorted(by_name), ["local-tools", "remote-api"])
            local = by_name["local-tools"]
            self.assertEqual(local.client, "Codex CLI")
            self.assertEqual(local.transport, "stdio")
            self.assertEqual(local.command, "python")
            self.assertEqual(local.args, ["server.py", "--verbose"])
            self.assertEqual(local.env, {"API_KEY": "x", "DEBUG": "1"})
            remote = by_name["remote-api"]
            self.assertEqual(remote.client, "Codex CLI")
            self.assertEqual(remote.transport, "http")
            self.assertEqual(remote.url, "https://example.com/mcp")

    def test_malformed_toml_produces_warning_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "config.toml", "[mcp_servers.broken\n")
            servers, warning = parse_toml_config_file(path)
            self.assertEqual(servers, [])
            self.assertIsNotNone(warning)
            self.assertIn("could not parse", warning.message)


_HTTP_SERVER_SOURCE = '''
import sys

import uvicorn

try:  # mcp SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("url-test")


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


class TestUrlAudit(unittest.TestCase):
    """--url against a locally spawned, auth-gated streamable-HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="mcp-bloat-url-test-")
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

    def _run_cli_json(self, argv: list[str]) -> dict:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(argv)
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    def test_url_without_header_is_rejected_by_server(self):
        report = self._run_cli_json(["--url", self.url, "--json"])
        (server,) = report["servers"]
        self.assertEqual(server["client"], "Remote URL")
        self.assertEqual(server["name"], "127.0.0.1")
        self.assertFalse(server["ok"])

    def test_url_with_header_and_name(self):
        report = self._run_cli_json(
            [
                "--url",
                self.url,
                "--name",
                "authed-demo",
                "--header",
                "Authorization: Bearer test-token",
                "--json",
            ]
        )
        (server,) = report["servers"]
        self.assertTrue(server["ok"])
        self.assertEqual(server["name"], "authed-demo")
        self.assertEqual(server["tool_count"], 2)
        self.assertGreater(server["total_tokens"], 0)
        self.assertEqual(report["total_tokens"], server["total_tokens"])


if __name__ == "__main__":
    unittest.main()
