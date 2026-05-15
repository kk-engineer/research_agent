from __future__ import annotations

import time
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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

    if settings.verbosity_level >= 1:
        console.print(
            f"  🔍 [bold {Theme.SEARCH}]{provider}[/bold {Theme.SEARCH}] "
            f"\"{query[:200]}\"",
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

    record = SearchRecord(
        provider=provider,
        query=query[:120],
        depth=depth,
        max_results=max_results,
        latency=time.monotonic() - start,
        result_count=result_count,
        success=success,
    )
    metrics.record_search(record)

    if settings.verbosity_level >= 2:
        if results:
            result_table = Table(box=box.SIMPLE, show_edge=False)
            result_table.add_column("#", style="dim", width=3)
            result_table.add_column("Title", style="bold", width=40)
            result_table.add_column("URL", style="dim", width=60)

            for i, r in enumerate(results[:8], 1):
                result_table.add_row(
                    str(i),
                    (r.title or "")[:60],
                    (r.url or "")[:60],
                )
            if result_count > 8:
                result_table.add_row("…", f"+{result_count - 8} more", "")

            console.print(
                make_panel(
                    result_table,
                    title=f"[bold {Theme.PANEL_SEARCH}]🔍 Search Results — {provider} ({result_count} results)[/bold {Theme.PANEL_SEARCH}]",
                    border_style=Theme.PANEL_SEARCH,
                )
            )
        else:
            console.print(
                make_panel(
                    Text("No results returned", style=f"bold {Theme.WARNING}"),
                    title=f"[bold {Theme.PANEL_WARN}]🔍 Search: {provider} (empty)[/bold {Theme.PANEL_WARN}]",
                    border_style=Theme.PANEL_WARN,
                )
            )
    elif settings.verbosity_level >= 1:
        console.print(
            f"  [green]✔[/green] {provider} → {result_count} results"
        )
