import json
import tempfile
import unittest
from pathlib import Path

from mcp_bloat.discovery import discover, load_config_file, parse_config_file


def _write(tmp: str, name: str, content: str) -> Path:
    path = Path(tmp) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestParseConfigFile(unittest.TestCase):
    def test_stdio_and_http_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp,
                "mcp.json",
                json.dumps(
                    {
                        "mcpServers": {
                            "local": {
                                "command": "python",
                                "args": ["server.py"],
                                "env": {"API_KEY": "x"},
                            },
                            "remote": {"url": "https://example.com/mcp"},
                            "incomplete": {"note": "no command or url"},
                        }
                    }
                ),
            )
            servers, warning = parse_config_file(path, client="Test")
            self.assertIsNone(warning)
            self.assertEqual([s.name for s in servers], ["local", "remote"])
            local, remote = servers
            self.assertEqual(local.transport, "stdio")
            self.assertEqual(local.command, "python")
            self.assertEqual(local.args, ["server.py"])
            self.assertEqual(local.env, {"API_KEY": "x"})
            self.assertEqual(local.client, "Test")
            self.assertEqual(remote.transport, "http")
            self.assertEqual(remote.url, "https://example.com/mcp")

    def test_vscode_servers_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp,
                "mcp.json",
                json.dumps({"servers": {"only": {"command": "node", "args": []}}}),
            )
            servers, warning = parse_config_file(path, key="servers")
            self.assertIsNone(warning)
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0].command, "node")

    def test_malformed_json_produces_warning_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "broken.json", "{not valid json")
            servers, warning = parse_config_file(path)
            self.assertEqual(servers, [])
            self.assertIsNotNone(warning)
            self.assertIn("could not parse", warning.message)

    def test_missing_file_is_quiet(self):
        servers, warning = parse_config_file(Path("/nonexistent/nope.json"))
        self.assertEqual(servers, [])
        self.assertIsNone(warning)

    def test_load_config_file_reports_missing_file(self):
        result = load_config_file("/nonexistent/nope.json")
        self.assertEqual(result.servers, [])
        self.assertEqual(len(result.warnings), 1)


class TestDiscover(unittest.TestCase):
    def test_discovers_claude_code_and_cursor_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            cwd = Path(tmp) / "proj"
            _write(
                tmp,
                "home/.claude.json",
                json.dumps(
                    {"mcpServers": {"a": {"command": "python", "args": []}}}
                ),
            )
            _write(
                tmp,
                "proj/.cursor/mcp.json",
                json.dumps(
                    {"mcpServers": {"b": {"url": "http://localhost:9000/mcp"}}}
                ),
            )
            result = discover(home=home, cwd=cwd)
            names = sorted(s.name for s in result.servers)
            self.assertEqual(names, ["a", "b"])
            self.assertEqual(len(result.files_found), 2)


if __name__ == "__main__":
    unittest.main()
