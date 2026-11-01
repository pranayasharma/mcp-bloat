"""mcp-bloat command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import urlparse

from rich.console import Console

from . import __version__
from .audit import DEFAULT_TIMEOUT, audit_all
from .demo import run_demo
from .discovery import ServerEntry, discover, load_config_file, searched_locations
from .models import DEFAULT_CONTEXT_WINDOW, ModelInfo, resolve_model
from .report import grand_total, render_json, render_terminal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-bloat",
        description=(
            "Audit how many tokens your MCP servers eat from your context "
            "window — before your agent pays for them."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="audit three generated demo servers instead of your real configs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the terminal report",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        metavar="N",
        help="exit with code 1 if the total token count exceeds N (for CI)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="audit one specific MCP config file instead of auto-discovery",
    )
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help="audit one remote MCP server directly at this streamable-HTTP URL",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="display label for the --url server (default: the URL's host)",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="HEADER",
        help=(
            'extra HTTP header for --url, as "Name: Value" '
            '(e.g. "Authorization: Bearer TOKEN"); may be given more than once'
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"per-server connection timeout (default {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help=(
            "compute percentages against this model's context window "
            '(e.g. --model gpt-5, --model claude-sonnet-4-5); looked up '
            "live from OpenRouter/LiteLLM, no API key needed"
        ),
    )
    parser.add_argument(
        "--window",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "compute percentages against a context window of N tokens; "
            "overrides --model (default window: "
            f"{DEFAULT_CONTEXT_WINDOW:,})"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"mcp-bloat {__version__}"
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if number <= 0:
        raise argparse.ArgumentTypeError("window size must be positive")
    return number


def _parse_headers(
    values: list[str] | None, parser: argparse.ArgumentParser
) -> dict[str, str]:
    """Turn repeatable --header "Name: Value" options into a mapping."""
    headers: dict[str, str] = {}
    for value in values or []:
        name, sep, val = value.partition(":")
        if not sep or not name.strip():
            parser.error(f"invalid --header {value!r}: expected 'Name: Value'")
        headers[name.strip()] = val.strip()
    return headers


def _resolve_window(
    args: argparse.Namespace, err_console: Console
) -> tuple[int, ModelInfo | None] | int:
    """Resolve the context window from --window / --model.

    Returns (context_window, model) on success, or an exit code to
    return immediately (2 when --model names nothing we can find).
    """
    if args.window is not None:
        return args.window, None
    if not args.model:
        return DEFAULT_CONTEXT_WINDOW, None

    resolution = resolve_model(args.model)
    if resolution.info is not None:
        return resolution.info.context_length, resolution.info
    if not resolution.data_available:
        err_console.print(
            "[yellow]warning:[/yellow] could not look up model data "
            "(offline and no cached copy), and "
            f"{args.model!r} is not in the built-in snapshot; falling "
            f"back to the default {DEFAULT_CONTEXT_WINDOW:,}-token window"
        )
        return DEFAULT_CONTEXT_WINDOW, None

    err_console.print(
        f"[red]error:[/red] no model matching {args.model!r} was found."
    )
    if resolution.candidates:
        err_console.print("Closest matches:")
        for candidate in resolution.candidates:
            err_console.print(
                f"  {candidate.id} — {candidate.context_length:,} tokens"
            )
        err_console.print("Re-run with one of those names, or use --window N.")
    else:
        err_console.print(
            "Check the spelling, or use --window N to set the window "
            "size directly."
        )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()
    err_console = Console(stderr=True)

    resolved = _resolve_window(args, err_console)
    if isinstance(resolved, int):
        return resolved
    context_window, model = resolved

    if args.demo:
        results, warnings = run_demo(timeout=args.timeout)
    elif args.url:
        if args.config:
            err_console.print(
                "[yellow]warning:[/yellow] --config is ignored when --url "
                "is given"
            )
        entry = ServerEntry(
            name=args.name or urlparse(args.url).hostname or args.url,
            source=args.url,
            client="Remote URL",
            transport="http",
            url=args.url,
            headers=_parse_headers(args.header, parser),
        )
        warnings = []
        results = asyncio.run(audit_all([entry], timeout=args.timeout))
    else:
        if args.name or args.header:
            err_console.print(
                "[yellow]warning:[/yellow] --name/--header only apply "
                "together with --url; ignoring them"
            )
        if args.config:
            discovered = load_config_file(args.config)
        else:
            discovered = discover()
        warnings = discovered.warnings
        if not discovered.servers:
            if args.json:
                print(
                    render_json(
                        [],
                        warnings,
                        context_window=context_window,
                        model=model,
                    )
                )
            else:
                for warning in warnings:
                    err_console.print(
                        f"[yellow]warning:[/yellow] {warning.source}: "
                        f"{warning.message}"
                    )
                console.print("No MCP servers found in any known config file.")
                console.print("Searched:")
                for location in searched_locations():
                    console.print(f"  {location}")
                console.print(
                    "Tip: point --config at a config file, or try --demo."
                )
            return 0
        results = asyncio.run(audit_all(discovered.servers, timeout=args.timeout))

    total = grand_total(results)
    if args.json:
        print(
            render_json(
                results,
                warnings,
                context_window=context_window,
                model=model,
            )
        )
    else:
        render_terminal(
            results,
            warnings,
            console=console,
            context_window=context_window,
            model=model,
        )

    if args.budget is not None and total > args.budget:
        err_console.print(
            f"[red]budget exceeded:[/red] {total:,} tokens > budget "
            f"{args.budget:,}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
