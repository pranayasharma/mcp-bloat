"""Terminal (rich) and JSON renderings of an audit run."""

from __future__ import annotations

import difflib
import json
from collections import OrderedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .audit import ServerAudit, ToolInfo
from .discovery import DiscoveryWarning
from .models import DEFAULT_CONTEXT_WINDOW, ModelInfo, short_display_name
from .tokens import counting_method

# Backwards-compatible alias: the default window used when no --model or
# --window is given.
CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW
BLOAT_THRESHOLD = 10_000  # a single server above this gets flagged


def grand_total(results: list[ServerAudit]) -> int:
    return sum(r.total_tokens for r in results)


def sorted_tools(audit: ServerAudit) -> list[ToolInfo]:
    """A server's tools, most expensive first."""
    return sorted(audit.tools, key=lambda t: t.tokens, reverse=True)


def percent_of_server(tool_tokens: int, audit: ServerAudit) -> float:
    """A tool's share of its server's tool-token total, as a percentage.

    The per-tool counts sum to the server's tools_tokens by
    construction, so the shares of one server's tools add up to 100%.
    """
    if audit.tools_tokens <= 0:
        return 0.0
    return 100.0 * tool_tokens / audit.tools_tokens


def most_expensive_tool(
    results: list[ServerAudit],
) -> tuple[ServerAudit, ToolInfo] | None:
    """The single priciest tool across all successfully audited servers."""
    best: tuple[ServerAudit, ToolInfo] | None = None
    for result in results:
        if not result.ok:
            continue
        for info in result.tools:
            if best is None or info.tokens > best[1].tokens:
                best = (result, info)
    return best


def percent_of_context(
    tokens: int, context_window: int = DEFAULT_CONTEXT_WINDOW
) -> float:
    if context_window <= 0:
        return 0.0
    return 100.0 * tokens / context_window


# ---------------------------------------------------------------------------
# Single-tool lookup (--tool NAME): the audit runs as usual, then the
# report focuses on just the tool(s) matching the requested name.
# ---------------------------------------------------------------------------


def find_tool_matches(
    results: list[ServerAudit], query: str
) -> tuple[list[tuple[ServerAudit, ToolInfo]], list[str]]:
    """Locate one tool by name across all successfully audited servers.

    Returns (matches, suggestions). Matching tiers: an exact
    case-insensitive name match wins; failing that, case-insensitive
    substring matches. When neither yields anything, ``matches`` is
    empty and ``suggestions`` holds the closest tool names that do
    exist (difflib, at most 5).
    """
    pairs = [
        (audit, info)
        for audit in results
        if audit.ok
        for info in audit.tools
    ]
    needle = query.lower()
    exact = [pair for pair in pairs if pair[1].name.lower() == needle]
    if exact:
        return exact, []
    substring = [pair for pair in pairs if needle in pair[1].name.lower()]
    if substring:
        return substring, []
    names = list(dict.fromkeys(info.name for _audit, info in pairs))
    suggestions = difflib.get_close_matches(query, names, n=5, cutoff=0.6)
    if not suggestions:
        lowered = {name.lower(): name for name in names}
        suggestions = [
            lowered[match]
            for match in difflib.get_close_matches(
                needle, list(lowered), n=5, cutoff=0.6
            )
        ]
    return [], suggestions


def _tool_match_dict(
    audit: ServerAudit, info: ToolInfo, context_window: int
) -> dict:
    return {
        "name": info.name,
        "server": audit.entry.name,
        "client": audit.entry.client,
        "source": audit.entry.source,
        "tokens": info.tokens,
        "percent_of_server": round(percent_of_server(info.tokens, audit), 2),
        "percent_of_context": round(
            percent_of_context(info.tokens, context_window), 2
        ),
        "description_tokens": info.description_tokens,
        "schema_tokens": info.schema_tokens,
    }


def tool_to_json_dict(
    results: list[ServerAudit],
    query: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> dict:
    """The --tool report as a JSON-ready dict (matches may be empty)."""
    matches, _suggestions = find_tool_matches(results, query)
    return {
        "tool": query,
        "matches": [
            _tool_match_dict(audit, info, context_window)
            for audit, info in matches
        ],
    }


def render_tool_json(
    results: list[ServerAudit],
    query: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> str:
    return json.dumps(
        tool_to_json_dict(results, query, context_window), indent=2
    )


def render_tool_terminal(
    matches: list[tuple[ServerAudit, ToolInfo]],
    query: str,
    warnings: list[DiscoveryWarning] | None = None,
    console: Console | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
) -> None:
    """Focused terminal report for the tool(s) matching ``query``."""
    console = console or Console()

    for warning in warnings or []:
        console.print(
            f"[yellow]warning:[/yellow] {warning.source}: {warning.message}"
        )

    table = Table(
        title=f"Tool lookup: {query}",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Tool")
    table.add_column("Server")
    table.add_column("Source")
    table.add_column("Tokens", justify="right")
    table.add_column("% of server", justify="right")
    table.add_column(f"% of {_short_window(context_window)} ctx", justify="right")
    table.add_column("Cost split")
    for audit, info in matches:
        table.add_row(
            info.name,
            audit.entry.name,
            f"{audit.entry.client} — {audit.entry.source}",
            _fmt(info.tokens),
            f"{percent_of_server(info.tokens, audit):.1f}%",
            f"{percent_of_context(info.tokens, context_window):.1f}%",
            f"description ≈ {_fmt(info.description_tokens)}"
            f" · schema ≈ {_fmt(info.schema_tokens)}",
        )
    console.print(table)
    console.print()

    summary = Text()
    count = len(matches)
    summary.append(
        f"{count} match{'es' if count != 1 else ''} for {query!r} "
        f"across {len({id(audit) for audit, _info in matches})} server(s).\n"
    )
    summary.append(
        "Cost split: description counts the tool's name + description "
        "text; schema counts its inputSchema. The two parts plus the "
        "JSON structure roughly add up to the total.\n"
    )
    summary.append(f"Counting method: {counting_method()}")
    if model is not None:
        summary.append(
            f"\nModel: {model.id} ({model.name}) — window "
            f"{_fmt(context_window)} tokens, via {model.source}"
        )
    console.print(
        Panel(summary, title="mcp-bloat tool lookup", title_align="left", expand=False)
    )


def _short_window(context_window: int) -> str:
    """Compact window label for table headers: 200k, 400k, 1M, 1.31M."""
    if context_window >= 1_000_000:
        value = context_window / 1_000_000
        return f"{value:.0f}M" if value == int(value) else f"{value:.2f}M"
    return f"{context_window / 1000:.0f}k"


def to_json_dict(
    results: list[ServerAudit],
    warnings: list[DiscoveryWarning] | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
    frontier: list[ModelInfo] | None = None,
) -> dict:
    total = grand_total(results)
    top = most_expensive_tool(results)
    report = {
        "counting_method": counting_method(),
        "model": model.id if model else None,
        "context_window": context_window,
        "total_tokens": total,
        "percent_of_context": round(percent_of_context(total, context_window), 2),
        "most_expensive_tool": (
            {
                "server": top[0].entry.name,
                "tool": top[1].name,
                "tokens": top[1].tokens,
            }
            if top is not None
            else None
        ),
        "servers": [
            {
                "name": r.entry.name,
                "client": r.entry.client,
                "source": r.entry.source,
                "transport": r.entry.transport,
                "ok": r.ok,
                "error": r.error,
                "tool_count": r.tool_count,
                "tools": [
                    {
                        "name": t.name,
                        "tokens": t.tokens,
                        "percent_of_server": round(
                            percent_of_server(t.tokens, r), 2
                        ),
                        "description_tokens": t.description_tokens,
                        "schema_tokens": t.schema_tokens,
                    }
                    for t in sorted_tools(r)
                ],
                "tools_tokens": r.tools_tokens,
                "prompt_count": r.prompt_count,
                "prompts_tokens": r.prompts_tokens,
                "resource_count": r.resource_count,
                "resources_tokens": r.resources_tokens,
                "total_tokens": r.total_tokens,
                "percent_of_context": round(
                    percent_of_context(r.total_tokens, context_window), 2
                ),
            }
            for r in results
        ],
        "warnings": [
            {"source": w.source, "message": w.message} for w in (warnings or [])
        ],
    }
    if frontier is not None:
        # Present exactly when the caller ran in default mode (no
        # --model/--window): the audit total lined up against several
        # frontier models' windows at once. Absent in single-window mode.
        report["frontier"] = [
            {
                "model": info.id,
                "name": info.name,
                "context_window": info.context_length,
                "percent_of_context": round(
                    percent_of_context(total, info.context_length), 2
                ),
            }
            for info in frontier
        ]
    return report


def render_json(
    results: list[ServerAudit],
    warnings: list[DiscoveryWarning] | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
    frontier: list[ModelInfo] | None = None,
) -> str:
    return json.dumps(
        to_json_dict(results, warnings, context_window, model, frontier),
        indent=2,
    )


def _fmt(n: int) -> str:
    return f"{n:,}"


def _render_tool_breakdown(console: Console, audit: ServerAudit) -> None:
    """One server's tools, most expensive first."""
    table = Table(
        title=f"{audit.entry.name} — tools by cost",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("Tool")
    table.add_column("Tokens", justify="right")
    table.add_column("% of server", justify="right")
    for info in sorted_tools(audit):
        table.add_row(
            info.name,
            _fmt(info.tokens),
            f"{percent_of_server(info.tokens, audit):.1f}%",
        )
    console.print(table)
    console.print()


def render_terminal(
    results: list[ServerAudit],
    warnings: list[DiscoveryWarning] | None = None,
    console: Console | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
    frontier: list[ModelInfo] | None = None,
    show_tools: bool = True,
) -> None:
    console = console or Console()

    for warning in warnings or []:
        console.print(
            f"[yellow]warning:[/yellow] {warning.source}: {warning.message}"
        )

    # Group by config file, preserving discovery order.
    by_source: "OrderedDict[str, list[ServerAudit]]" = OrderedDict()
    for result in results:
        by_source.setdefault(result.entry.source, []).append(result)

    for source, group in by_source.items():
        client = group[0].entry.client
        table = Table(
            title=f"{client} — {source}",
            title_justify="left",
            header_style="bold",
        )
        table.add_column("Server")
        table.add_column("Transport")
        table.add_column("Tools", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column(f"% of {_short_window(context_window)} ctx", justify="right")
        table.add_column("Status")
        for r in group:
            if r.ok:
                status = Text("ok", style="green")
                if r.total_tokens > BLOAT_THRESHOLD:
                    status = Text("⚠ BLOAT", style="bold red")
                tokens_text = Text(
                    _fmt(r.total_tokens),
                    style="bold red" if r.total_tokens > BLOAT_THRESHOLD else "",
                )
                table.add_row(
                    r.entry.name,
                    r.entry.transport,
                    str(r.tool_count),
                    tokens_text,
                    f"{percent_of_context(r.total_tokens, context_window):.1f}%",
                    status,
                )
            else:
                table.add_row(
                    r.entry.name,
                    r.entry.transport,
                    "—",
                    "—",
                    "—",
                    Text(f"error: {r.error}", style="red"),
                )
        console.print(table)
        console.print()
        if show_tools:
            for r in group:
                if r.ok and r.tools:
                    _render_tool_breakdown(console, r)

    total = grand_total(results)
    ok_count = sum(1 for r in results if r.ok)
    failed = len(results) - ok_count
    bloated = [r for r in results if r.ok and r.total_tokens > BLOAT_THRESHOLD]

    summary = Text()
    summary.append("Total: ", style="bold")
    summary.append(f"{_fmt(total)} tokens")
    if model is not None:
        window_phrase = f"{model.id}'s {_fmt(context_window)}-token window"
    else:
        window_phrase = f"a {_fmt(context_window)}-token context window"
    summary.append(
        f"  ({percent_of_context(total, context_window):.1f}% of "
        f"{window_phrase})\n"
    )
    if frontier:
        # Default mode only: line the same total up against the windows
        # of several frontier models at once.
        summary.append(f"Of your {_fmt(total)} tokens, that's:\n")
        for info in frontier:
            summary.append(
                f"  {percent_of_context(total, info.context_length):.1f}% "
                f"of {short_display_name(info)}'s "
                f"{_fmt(info.context_length)} window\n"
            )
    summary.append(f"Servers audited: {ok_count}")
    if failed:
        summary.append(f"   ·   failed: {failed}", style="red")
    summary.append(f"\nCounting method: {counting_method()}")
    if model is not None:
        summary.append(
            f"\nModel: {model.id} ({model.name}) — window "
            f"{_fmt(context_window)} tokens, via {model.source}"
        )
    if show_tools:
        top = most_expensive_tool(results)
        if top is not None:
            server, info = top
            summary.append(
                f"\nMost expensive single tool: {info.name} "
                f"({server.entry.name}) — {_fmt(info.tokens)} tokens"
            )
    if bloated:
        names = ", ".join(
            f"{r.entry.name} ({_fmt(r.total_tokens)})" for r in bloated
        )
        summary.append(
            f"\n⚠ Over {_fmt(BLOAT_THRESHOLD)} tokens on its own: {names}",
            style="bold red",
        )
    console.print(
        Panel(summary, title="mcp-bloat summary", title_align="left", expand=False)
    )
