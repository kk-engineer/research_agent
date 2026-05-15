from __future__ import annotations

import time
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.console import Group
from rich import box

from src.config import settings
from src.rich_console import console, Theme, make_panel, make_key_value_table, syntax_block
from src.metrics import metrics, SearchRecord


def log_search_start(
    provider: str,
    query: str,
    max_results: int = 10,
    depth: str = "advanced",
) -> float:
    start = time.monotonic()

    show_full = settings.show_search_details

    if show_full:
        console.print(
            make_panel(
                Group(
                    make_key_value_table([
                        ("Provider", provider),
                        ("Query", query),
                        ("Max results", str(max_results)),
                        ("Depth", depth),
                    ]),
                ),
                title="[bold]🔍 Web Search[/bold]",
                border_style=Theme.PANEL_SEARCH,
            )
        )
    else:
        console.print(
            f"  🔍 [bold {Theme.SEARCH}]{provider}[/bold {Theme.SEARCH}] "
            f"\"{query[:120]}\"",
        )

    return start


def log_search_end(
    provider: str,
    query: str,
    start: float,
    results: list[Any],
    success: bool = True,
    max_results: int = 10,
    depth: str = "advanced",
) -> None:
    result_count = len(results)
    elapsed = time.monotonic() - start

    record = SearchRecord(
        provider=provider,
        query=query[:200],
        depth=depth,
        max_results=max_results,
        latency=elapsed,
        result_count=result_count,
        success=success,
    )
    metrics.record_search(record)

    show_full = settings.show_search_details

    if show_full:
        content_parts = [
            make_key_value_table([
                ("Provider", provider),
                ("Results", str(result_count)),
                ("Time", f"{elapsed:.2f}s"),
            ]),
        ]

        if results:
            result_rows = []
            for i, r in enumerate(results, 1):
                title = r.title or "No title"
                url = r.url or "No URL"
                snippet = (r.snippet or "")[:200]
                result_rows.append(
                    Panel(
                        Text(f"{title}\n{url}\n{snippet}", style=Theme.INFO),
                        title=f"[bold]#{i}[/bold]",
                        border_style="dim",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            if result_rows:
                content_parts.append(Text(""))
                content_parts.append(Group(*result_rows))
        else:
            content_parts.append(Text("No results returned", style=f"bold {Theme.WARNING}"))

        console.print(
            make_panel(
                Group(*content_parts),
                title=f"[bold {Theme.PANEL_SEARCH}]🔍 Search Results — {provider} ({result_count} results)[/bold {Theme.PANEL_SEARCH}]",
                border_style=Theme.PANEL_SEARCH,
            )
        )
    else:
        console.print(
            f"  [green]✔[/green] {provider} → {result_count} results  ({elapsed:.2f}s)"
        )
