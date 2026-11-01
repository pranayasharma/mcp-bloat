"""Terminal (rich) and JSON renderings of an audit run."""

from __future__ import annotations

import json
from collections import OrderedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .audit import ServerAudit
from .discovery import DiscoveryWarning
from .models import DEFAULT_CONTEXT_WINDOW, ModelInfo
from .tokens import counting_method

# Backwards-compatible alias: the default window used when no --model or
# --window is given.
CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW
BLOAT_THRESHOLD = 10_000  # a single server above this gets flagged


def grand_total(results: list[ServerAudit]) -> int:
    return sum(r.total_tokens for r in results)


def percent_of_context(
    tokens: int, context_window: int = DEFAULT_CONTEXT_WINDOW
) -> float:
    if context_window <= 0:
        return 0.0
    return 100.0 * tokens / context_window


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
) -> dict:
    total = grand_total(results)
    return {
        "counting_method": counting_method(),
        "model": model.id if model else None,
        "context_window": context_window,
        "total_tokens": total,
        "percent_of_context": round(percent_of_context(total, context_window), 2),
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
                    {"name": t.name, "tokens": t.tokens} for t in r.tools
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


def render_json(
    results: list[ServerAudit],
    warnings: list[DiscoveryWarning] | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
) -> str:
    return json.dumps(
        to_json_dict(results, warnings, context_window, model), indent=2
    )


def _fmt(n: int) -> str:
    return f"{n:,}"


def render_terminal(
    results: list[ServerAudit],
    warnings: list[DiscoveryWarning] | None = None,
    console: Console | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    model: ModelInfo | None = None,
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
    summary.append(f"Servers audited: {ok_count}")
    if failed:
        summary.append(f"   ·   failed: {failed}", style="red")
    summary.append(f"\nCounting method: {counting_method()}")
    if model is not None:
        summary.append(
            f"\nModel: {model.id} ({model.name}) — window "
            f"{_fmt(context_window)} tokens, via {model.source}"
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
