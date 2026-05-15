from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.rich_console import console, Theme, styled, syntax_block


@dataclass
class SpanEvent:
    timestamp: float
    label: str
    category: str
    duration: Optional[float] = None
    details: dict = field(default_factory=dict)
    level: int = 0


_CATEGORY_ICONS = {
    "llm": "🤖",
    "search": "🔍",
    "extract": "📄",
    "dedup": "🧹",
    "contradiction": "⚡",
    "synthesize": "📝",
    "analyze": "🔬",
    "plan": "📋",
    "clarify": "💬",
    "fetch": "🌐",
    "score": "📊",
    "general": "▸",
}


class Tracer:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._events: list[SpanEvent] = []
        self._level: int = 0

    def start(self) -> None:
        self._start = time.monotonic()
        self._events.clear()
        self._level = 0

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    def event(
        self,
        label: str,
        category: str = "general",
        details: dict | None = None,
    ) -> None:
        if self._start is None:
            return
        self._events.append(
            SpanEvent(
                timestamp=self.elapsed,
                label=label,
                category=category,
                details=details or {},
                level=self._level,
            )
        )

    def complete_event(
        self,
        label: str,
        category: str = "general",
        duration: float = 0.0,
        details: dict | None = None,
    ) -> None:
        if self._start is None:
            return
        self._events.append(
            SpanEvent(
                timestamp=max(0.0, self.elapsed - duration),
                label=label,
                category=category,
                duration=duration,
                details=details or {},
                level=self._level,
            )
        )

    def push_level(self) -> None:
        self._level += 1

    def pop_level(self) -> None:
        self._level = max(0, self._level - 1)

    @property
    def total_llm_time(self) -> float:
        return sum(
            e.duration or 0.0 for e in self._events if e.category == "llm"
        )

    @property
    def total_search_time(self) -> float:
        return sum(
            e.duration or 0.0 for e in self._events if e.category == "search"
        )

    def render_panel(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(width=4)
        table.add_column(width=10)
        table.add_column()
        table.add_column(width=10, max_width=10)

        for ev in self._events:
            icon = _CATEGORY_ICONS.get(ev.category, "▸")
            ts = f"[{ev.timestamp:>7.3f}]"
            dur = f"{ev.duration:.2f}s" if ev.duration else ""
            indent = "  " * ev.level
            label_text = f"{indent}{ev.label}"
            table.add_row(
                Text(icon, style=f"bold {Theme.HIGHLIGHT}"),
                Text(ts, style=f"bold {Theme.TIME}"),
                Text(label_text, style=Theme.INFO),
                Text(dur, style=f"bold {Theme.SUCCESS}" if ev.duration else Theme.DIM),
            )

        return Panel(
            table,
            title=f"[bold {Theme.PANEL_TIMELINE}]Execution Timeline[/bold {Theme.PANEL_TIMELINE}]",
            border_style=Theme.PANEL_TIMELINE,
            padding=(1, 1),
            box=box.ROUNDED,
        )


tracer = Tracer()
