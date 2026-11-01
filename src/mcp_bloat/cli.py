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
        "--version", action="version", version=f"mcp-bloat {__version__}"
    )
    return parser


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()
    err_console = Console(stderr=True)

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
                print(render_json([], warnings))
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
        print(render_json(results, warnings))
    else:
        render_terminal(results, warnings, console=console)

    if args.budget is not None and total > args.budget:
        err_console.print(
            f"[red]budget exceeded:[/red] {total:,} tokens > budget "
            f"{args.budget:,}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
