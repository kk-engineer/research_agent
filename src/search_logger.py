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

    console.print(
        make_panel(
            make_key_value_table([
                ("Provider", provider),
                ("Query", query),
                ("Max results", str(max_results)),
                ("Depth", depth),
            ]),
            title=f"[bold {Theme.PANEL_SEARCH_REQ}]🔍 Search Request — {provider}[/bold {Theme.PANEL_SEARCH_REQ}]",
            border_style=Theme.PANEL_SEARCH_REQ,
        )
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

    content_parts = [
        make_key_value_table([
            ("Provider", provider),
            ("Results", str(result_count)),
            ("Time", f"{elapsed:.2f}s"),
        ]),
    ]

    if results:
        result_panels = []
        for i, r in enumerate(results, 1):
            detail_text = Text.assemble(
                ("Title: ", "bold"), (f"{r.title or 'No title'}\n", ""),
                ("URL: ", "bold"), (f"{r.url or 'No URL'}\n", ""),
                ("Snippet: ", "bold"), (f"{(r.snippet or '')[:500]}\n", ""),
            )
            if getattr(r, 'content', None):
                content_str = r.content or ""
                if content_str:
                    detail_text.append_text(
                        Text.assemble(
                            ("Content: ", "bold"),
                            (f"{content_str[:800]}", "dim"),
                        )
                    )
            if getattr(r, 'source', None):
                detail_text.append_text(Text(f"\nSource: {r.source}", style="bold"))
            if getattr(r, 'published_date', None):
                detail_text.append_text(Text(f"\nPublished: {r.published_date}", style="dim"))

            result_panels.append(
                Panel(
                    detail_text,
                    title=f"[bold]#{i}[/bold]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        if result_panels:
            content_parts.append(Text(""))
            content_parts.append(Group(*result_panels))
    else:
        content_parts.append(Text("No results returned", style=f"bold {Theme.WARNING}"))

    console.print(
        make_panel(
            Group(*content_parts),
            title=(
                f"[bold {Theme.PANEL_SEARCH_RESP}]🔍 Search Response — {provider} "
                f"({result_count} results, {elapsed:.2f}s)[/bold {Theme.PANEL_SEARCH_RESP}]"
            ),
            border_style=Theme.PANEL_SEARCH_RESP,
        )
    )
