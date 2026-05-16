from __future__ import annotations

import time
from typing import Any

from rich.panel import Panel
from rich.text import Text
from rich import box

from src.config import settings
from src.monitoring.async_logger import get_async_logger
from src.monitoring.metrics import metrics, SearchRecord
from src.ui import events as _events
from src.ui.rich_console import console, Theme, make_panel, make_key_value_table


def log_search_start(
    provider: str,
    query: str,
    max_results: int = 10,
    depth: str = "advanced",
) -> float:
    start = time.monotonic()

    _events.event_bus.emit_kv(_events.EventType.SEARCH_STARTED, provider=provider, query=query[:80], max_results=max_results)

    if not _events.tui_mode:
        console.print(
            make_panel(
                make_key_value_table([
                    ("Provider", provider),
                    ("Query", query),
                    ("Max results", str(max_results)),
                    ("Depth", depth),
                ]),
                title=f"🔍 Search Request — {provider}",
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

    _events.event_bus.emit_kv(
        _events.EventType.SEARCH_COMPLETED,
        provider=provider,
        query=query[:80],
        results_count=result_count,
        latency=round(elapsed, 2),
        success=success,
    )

    get_async_logger().emit("search", {
        "provider": provider,
        "query": query[:200],
        "duration": round(elapsed, 3),
        "results": result_count,
        "success": success,
    })

    if not _events.tui_mode:
        console.print(
            make_panel(
                make_key_value_table([
                    ("Provider", provider),
                    ("Results", str(result_count)),
                    ("Time", f"{elapsed:.2f}s"),
                ]),
                title=f"🔍 Search Response — {provider} ({result_count} results, {elapsed:.2f}s)",
                border_style=Theme.PANEL_SEARCH_RESP,
            )
        )

    if not settings.show_search_details:
        return

    if not _events.tui_mode:
        if not results:
            console.print(
                Panel(
                    Text("No results returned", style=f"bold {Theme.WARNING}"),
                    title="Search Results",
                    border_style=Theme.PANEL_WARN,
                    box=box.ROUNDED,
                )
            )
            return

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

            console.print(
                Panel(
                    detail_text,
                    title=f"Result #{i}",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
