"""Connect to each discovered MCP server and measure its context cost.

For every server we perform a real MCP handshake, list its tools (plus
prompts and resources, which clients also surface to the model), and
count the tokens of the JSON serialization a client would send to the
model: name + description + inputSchema for each tool.

A server that fails to start, times out, or speaks garbage produces an
error result — it never crashes the whole audit.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from .discovery import ServerEntry
from .tokens import count_tokens

try:  # mcp SDK >= 1.x exposes this at the top level
    from mcp import ClientSession, StdioServerParameters
except ImportError:  # pragma: no cover
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters

from mcp.client.stdio import stdio_client

try:  # mcp SDK 2.x name
    from mcp.client.streamable_http import streamable_http_client as _http_client
except ImportError:  # mcp SDK 1.x name
    from mcp.client.streamable_http import streamablehttp_client as _http_client

try:  # factory for a pre-configured httpx client (SDK 1.x late / 2.x)
    from mcp.client.streamable_http import (
        create_mcp_http_client as _create_http_client,
    )
except ImportError:  # pragma: no cover
    _create_http_client = None  # type: ignore[assignment]

# How this SDK generation accepts request headers:
# - 2.x: streamable_http_client(url, http_client=<httpx client>) — headers
#   ride on a caller-supplied client built by create_mcp_http_client, and
#   the transport does not manage that client's lifecycle.
# - 1.x: streamablehttp_client(url, headers={...}) took them directly.
_HTTP_PARAMS = inspect.signature(_http_client).parameters
_HTTP_ACCEPTS_CLIENT = "http_client" in _HTTP_PARAMS
_HTTP_ACCEPTS_HEADERS = "headers" in _HTTP_PARAMS

DEFAULT_TIMEOUT = 20.0
_MAX_CONCURRENT = 5


@dataclass
class ToolInfo:
    name: str
    tokens: int
    # Cost split of the tool payload: the {name, description} part and
    # the inputSchema part, counted separately. The two parts plus the
    # JSON structure of the combined payload roughly add up to
    # ``tokens`` — close, not exact, so reports present them with "≈".
    description_tokens: int = 0
    schema_tokens: int = 0


@dataclass
class ServerAudit:
    entry: ServerEntry
    ok: bool = False
    error: str | None = None
    tools: list[ToolInfo] = field(default_factory=list)
    tools_tokens: int = 0
    prompts_tokens: int = 0
    resources_tokens: int = 0
    prompt_count: int = 0
    resource_count: int = 0

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def total_tokens(self) -> int:
        return self.tools_tokens + self.prompts_tokens + self.resources_tokens


def _describe_error(exc: BaseException) -> str:
    """Flatten ExceptionGroups (anyio) into a one-line message."""
    if isinstance(exc, BaseExceptionGroup):
        parts = [_describe_error(sub) for sub in exc.exceptions]
        return "; ".join(dict.fromkeys(parts))
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _model_payload(item: object) -> dict:
    """Best-effort wire-format dict for an MCP type across SDK versions."""
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump(by_alias=True, exclude_none=True, mode="json")
        except Exception:
            pass
    return {}


def _tool_tokens(tool: object) -> ToolInfo:
    data = _model_payload(tool)
    name = data.get("name") or getattr(tool, "name", "") or ""
    description = data.get("description") or getattr(tool, "description", "") or ""
    schema = data.get("inputSchema") or data.get("input_schema")
    if schema is None:
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {}
        )
    payload = {"name": name, "description": description, "inputSchema": schema}
    description_tokens = count_tokens(
        json.dumps(
            {"name": name, "description": description}, ensure_ascii=False
        )
    )
    schema_tokens = count_tokens(json.dumps(schema, ensure_ascii=False))
    return ToolInfo(
        name=str(name),
        tokens=count_tokens(json.dumps(payload, ensure_ascii=False)),
        description_tokens=description_tokens,
        schema_tokens=schema_tokens,
    )


def _item_tokens(item: object, keys: tuple[str, ...]) -> int:
    data = _model_payload(item)
    payload = {}
    for key in keys:
        value = data.get(key)
        if value is None:
            value = getattr(item, key, None)
        if value is not None:
            payload[key] = value
    return count_tokens(json.dumps(payload, ensure_ascii=False, default=str))


async def _collect(session: ClientSession, audit: ServerAudit) -> None:
    # Tools — the main event. A failure here fails the audit.
    cursor = None
    while True:
        if cursor is None:
            result = await session.list_tools()
        else:  # pragma: no cover - most servers fit in one page
            try:
                from mcp import types as _types

                result = await session.list_tools(
                    params=_types.PaginatedRequestParams(cursor=cursor)
                )
            except Exception:
                break
        for tool in result.tools:
            info = _tool_tokens(tool)
            audit.tools.append(info)
            audit.tools_tokens += info.tokens
        cursor = getattr(result, "nextCursor", None) or getattr(
            result, "next_cursor", None
        )
        if not cursor:
            break

    # Prompts and resources also land in the model's context in most
    # clients, but a server that doesn't implement them is normal —
    # count them if present, ignore failures.
    try:
        prompts = await session.list_prompts()
        audit.prompt_count = len(prompts.prompts)
        for prompt in prompts.prompts:
            audit.prompts_tokens += _item_tokens(
                prompt, ("name", "description", "arguments")
            )
    except Exception:
        pass
    try:
        resources = await session.list_resources()
        audit.resource_count = len(resources.resources)
        for resource in resources.resources:
            audit.resources_tokens += _item_tokens(
                resource, ("name", "description", "uri", "mimeType")
            )
    except Exception:
        pass


async def _audit_stdio(entry: ServerEntry, audit: ServerAudit) -> None:
    params = StdioServerParameters(
        command=entry.command or "",
        args=list(entry.args),
        env=dict(entry.env) if entry.env else None,
    )
    devnull = open(os.devnull, "wb")
    try:
        async with stdio_client(params, errlog=devnull) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await _collect(session, audit)
    finally:
        devnull.close()


def _unpack_streams(streams: object) -> tuple:
    """streamable http clients yield (read, write[, get_session_id])."""
    try:
        read, write, *_rest = streams  # type: ignore[misc]
        return read, write
    except (TypeError, ValueError):
        pass
    read = getattr(streams, "read_stream", None) or getattr(streams, "read", None)
    write = getattr(streams, "write_stream", None) or getattr(streams, "write", None)
    if read is None or write is None:
        raise RuntimeError("could not unpack HTTP transport streams")
    return read, write


@asynccontextmanager
async def _http_streams(
    url: str, headers: dict[str, str] | None = None
) -> AsyncIterator[tuple]:
    """Open the streamable-HTTP transport, with headers if requested."""
    if headers and _HTTP_ACCEPTS_CLIENT and _create_http_client is not None:
        # SDK 2.x: headers ride on a caller-supplied httpx client. The
        # transport leaves a supplied client's lifecycle to us, so we
        # own it here.
        async with _create_http_client(headers=headers) as http_client:
            async with _http_client(url, http_client=http_client) as streams:
                yield streams
    elif headers and _HTTP_ACCEPTS_HEADERS:
        # SDK 1.x accepted a headers mapping directly.
        async with _http_client(url, headers=headers) as streams:
            yield streams
    else:
        async with _http_client(url) as streams:
            yield streams


async def _audit_http(entry: ServerEntry, audit: ServerAudit) -> None:
    headers = entry.headers or None
    async with _http_streams(entry.url or "", headers) as streams:
        read, write = _unpack_streams(streams)
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _collect(session, audit)


async def _audit_inner(entry: ServerEntry, audit: ServerAudit) -> None:
    if entry.transport == "http":
        await _audit_http(entry, audit)
    else:
        await _audit_stdio(entry, audit)


async def audit_server(entry: ServerEntry, timeout: float = DEFAULT_TIMEOUT) -> ServerAudit:
    audit = ServerAudit(entry=entry)
    try:
        await asyncio.wait_for(_audit_inner(entry, audit), timeout=timeout)
        audit.ok = True
    except asyncio.TimeoutError:
        audit.error = f"timed out after {timeout:g}s"
    except Exception as exc:
        audit.error = _describe_error(exc)
    return audit


async def audit_all(
    entries: list[ServerEntry], timeout: float = DEFAULT_TIMEOUT
) -> list[ServerAudit]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _run(entry: ServerEntry) -> ServerAudit:
        async with semaphore:
            return await audit_server(entry, timeout=timeout)

    if not entries:
        return []
    return list(await asyncio.gather(*(_run(e) for e in entries)))
