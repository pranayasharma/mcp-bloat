"""Discovery of MCP server configurations across popular clients.

Looks for the config files used by Claude Code, Cursor, Claude Desktop,
Windsurf, and VS Code, and normalizes every server entry into a
:class:`ServerEntry`. Missing files are skipped silently; a file that
exists but cannot be parsed produces a warning instead of a crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Key under which each file stores its server map. VS Code is the odd one
# out: it uses "servers" where everyone else uses "mcpServers".
STD_KEY = "mcpServers"
VSCODE_KEY = "servers"


@dataclass
class ServerEntry:
    """One MCP server definition from a client config file."""

    name: str
    source: str  # path of the config file it came from
    client: str  # human label, e.g. "Claude Code"
    transport: str  # "stdio" or "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class DiscoveryWarning:
    source: str
    message: str


@dataclass
class DiscoveryResult:
    servers: list[ServerEntry] = field(default_factory=list)
    warnings: list[DiscoveryWarning] = field(default_factory=list)
    files_found: list[str] = field(default_factory=list)


def candidate_files(
    home: Path | None = None, cwd: Path | None = None
) -> list[tuple[str, Path, str]]:
    """All (client label, config path, server-map key) we know to look for."""
    home = home or Path.home()
    cwd = cwd or Path.cwd()
    return [
        ("Claude Code", home / ".claude.json", STD_KEY),
        ("Claude Code", cwd / ".mcp.json", STD_KEY),
        ("Cursor", home / ".cursor" / "mcp.json", STD_KEY),
        ("Cursor", cwd / ".cursor" / "mcp.json", STD_KEY),
        (
            "Claude Desktop",
            home / ".config" / "Claude" / "claude_desktop_config.json",
            STD_KEY,
        ),
        (
            "Claude Desktop",
            home
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json",
            STD_KEY,
        ),
        ("Windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", STD_KEY),
        ("VS Code", cwd / ".vscode" / "mcp.json", VSCODE_KEY),
    ]


def searched_locations() -> list[str]:
    """Display strings for every location discovery looks at."""
    return [str(path) for _client, path, _key in candidate_files()]


def _entry_from_raw(
    name: str, raw: object, source: str, client: str
) -> ServerEntry | None:
    if not isinstance(raw, dict):
        return None
    command = raw.get("command")
    url = raw.get("url")
    if command:
        args = raw.get("args") or []
        env = raw.get("env") or {}
        return ServerEntry(
            name=name,
            source=source,
            client=client,
            transport="stdio",
            command=str(command),
            args=[str(a) for a in args],
            env={str(k): str(v) for k, v in env.items()},
            raw=raw,
        )
    if url:
        return ServerEntry(
            name=name,
            source=source,
            client=client,
            transport="http",
            url=str(url),
            raw=raw,
        )
    return None


def parse_config_file(
    path: Path, client: str = "Config", key: str | None = None
) -> tuple[list[ServerEntry], DiscoveryWarning | None]:
    """Parse one config file into server entries.

    Returns (servers, warning). A missing file returns ([], None) so
    callers can probe speculatively; malformed JSON returns a warning.
    """
    if not path.is_file():
        return [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], DiscoveryWarning(
            source=str(path), message=f"could not parse config: {exc}"
        )
    if not isinstance(data, dict):
        return [], DiscoveryWarning(
            source=str(path), message="config root is not a JSON object"
        )

    keys = [key] if key else [STD_KEY, VSCODE_KEY]
    server_map: dict = {}
    for candidate in keys:
        value = data.get(candidate)
        if isinstance(value, dict):
            server_map = value
            break

    servers: list[ServerEntry] = []
    for name, raw in server_map.items():
        entry = _entry_from_raw(str(name), raw, str(path), client)
        if entry is not None:
            servers.append(entry)
    return servers, None


def load_config_file(path: str | Path) -> DiscoveryResult:
    """Load a single user-specified config file (for --config)."""
    path = Path(path).expanduser()
    result = DiscoveryResult()
    if not path.is_file():
        result.warnings.append(
            DiscoveryWarning(source=str(path), message="file does not exist")
        )
        return result
    servers, warning = parse_config_file(path, client="Custom config")
    result.servers.extend(servers)
    result.files_found.append(str(path))
    if warning is not None:
        result.warnings.append(warning)
    return result


def discover(
    home: Path | None = None, cwd: Path | None = None
) -> DiscoveryResult:
    """Scan every known client config location for MCP servers."""
    result = DiscoveryResult()
    seen_paths: set[str] = set()
    for client, path, key in candidate_files(home=home, cwd=cwd):
        resolved = str(path)
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not path.is_file():
            continue
        servers, warning = parse_config_file(path, client=client, key=key)
        result.files_found.append(resolved)
        result.servers.extend(servers)
        if warning is not None:
            result.warnings.append(warning)
    return result
