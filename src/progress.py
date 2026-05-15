from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


@dataclass
class PhaseRecord:
    name: str
    status: str
    duration: Optional[float] = None
    detail: str = ""


_RUNNING = "running"
_DONE = "done"
_FAILED = "failed"

_STL_HEADER = Style(bold=True, color="bright_cyan")
_STL_ACTIVE = Style(bold=True, color="bright_blue")
_STL_SUCCESS = Style(bold=True, color="green")
_STL_FAIL = Style(bold=True, color="red")
_STL_DIM = Style(dim=True, color="grey74")
_STL_MUTED = Style(color="grey58")
_STL_STAT_LABEL = Style(bold=True, color="bright_white")
_STL_STAT_VAL = Style(bold=True, color="bright_cyan")
_STL_TOOL = Style(bold=True, color="bright_yellow")

_ICONS = {
    _RUNNING: ("●", _STL_ACTIVE),
    _DONE: ("✔", _STL_SUCCESS),
    _FAILED: ("✘", _STL_FAIL),
}

_STATUS_LABELS = {
    _RUNNING: "RUN",
    _DONE: "DONE",
    _FAILED: "FAIL",
}

_LOG_STYLES = {
    "info": Style(color="grey74"),
    "done": _STL_SUCCESS,
    "warn": Style(color="bright_yellow"),
    "active": _STL_ACTIVE,
    "thought": Style(bold=True, color="bright_magenta"),
    "tool": _STL_TOOL,
    "llm": Style(color="bright_cyan"),
    "data": Style(color="grey74"),
}


class ResearchProgress:
    def __init__(self, console: Console, use_live: bool = True) -> None:
        self._console = console
        self._start = time.monotonic()
        self._phases: list[PhaseRecord] = []
        self._phase_timer: float = 0.0
        self._live: Optional[Live] = None
        self._use_live = use_live
        self._tokens_used: int = 0
        self._sources_found: int = 0
        self._claims_found: int = 0
        self._action_text: str = ""
        self._current_tool: str = ""
        self._step_timer: float = 0.0
        self._logs: deque = deque(maxlen=200)

    def start(self) -> None:
        if self._use_live:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=5,
            )
            self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        elapsed = time.monotonic() - self._start
        m, s = divmod(int(elapsed), 60)
        self._console.print(
            Panel(
                f"[bold green]Research complete[/bold green]  "
                f"⏱ {m:02d}:{s:02d}  "
                f"📊 {self._claims_found} claims  "
                f"🔗 {self._sources_found} sources",
                border_style="green",
            )
        )

    def phase(self, name: str, status: str = _RUNNING, detail: str = "") -> None:
        self._phases.append(PhaseRecord(name=name, status=status, detail=detail))
        self._phase_timer = time.monotonic()
        self._action_text = name
        self._refresh()

    def update_phase(self, name: str, status: str, detail: str = "") -> None:
        for p in reversed(self._phases):
            if p.name == name:
                p.status = status
                p.duration = time.monotonic() - self._phase_timer
                if detail:
                    p.detail = detail
                break
        self._refresh()

    def mark_done(self, name: str, detail: str = "") -> None:
        self.update_phase(name, _DONE, detail)

    def mark_failed(self, name: str, detail: str = "") -> None:
        self.update_phase(name, _FAILED, detail)

    def set_action(self, text: str) -> None:
        self._action_text = text
        self._step_timer = time.monotonic()
        self._refresh()

    def set_tool(self, tool: str) -> None:
        self._current_tool = tool
        self._step_timer = time.monotonic()
        self._refresh()

    def log(self, message: str, level: str = "info") -> None:
        self._logs.append((message, level))
        self._refresh()

    def set_tokens(self, n: int) -> None:
        self._tokens_used = n
        self._refresh()

    def add_tokens(self, n: int) -> None:
        self._tokens_used += n
        self._refresh()

    def add_sources(self, n: int) -> None:
        self._sources_found += n
        self._refresh()

    def add_claims(self, n: int) -> None:
        self._claims_found += n
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _completed_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _DONE)

    def _failed_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _FAILED)

    def _active_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _RUNNING)

    def _elapsed_str(self) -> str:
        t = time.monotonic() - self._start
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    def _step_elapsed(self) -> str:
        if self._step_timer == 0:
            return ""
        t = time.monotonic() - self._step_timer
        return f"{t:.1f}s"

    def _render(self) -> Panel:
        elapsed = self._elapsed_str()
        step_time = self._step_elapsed()

        table = Table.grid(padding=(0, 2))
        table.add_column(width=6)
        table.add_column(width=46)
        table.add_column(width=12)
        table.add_column(width=12)

        for p in self._phases:
            icon, style = _ICONS.get(p.status, ("●", _STL_MUTED))
            label = _STATUS_LABELS.get(p.status, "?  ")
            dur = f"[{p.duration:.1f}s]" if p.duration is not None else "..."
            name_st = style if p.status in (_DONE, _FAILED) else _STL_ACTIVE
            table.add_row(
                Text(f"{icon} {label}", style=style),
                Text(p.name[:46], style=name_st),
                Text(dur, style=_STL_DIM),
                Text(p.detail, style=_STL_STAT_VAL),
            )

        total_elapsed = time.monotonic() - self._start
        summary = (
            f"Total: {total_elapsed:.1f}s  "
            f"tok:{self._tokens_used:,}  "
            f"src:{self._sources_found}  "
            f"clm:{self._claims_found}"
        )
        table.add_row("", Text("...", style=_STL_HEADER), "", "")
        table.add_row(
            "",
            Text(summary, style=_STL_HEADER),
            "",
            "",
        )

        body: list = [table]

        if self._current_tool or self._action_text:
            parts = []
            if self._current_tool:
                parts.append(f"⚙ {self._current_tool}")
            if self._action_text:
                parts.append(f"▶ {self._action_text}")
            if step_time:
                parts.append(f"[{step_time}]")
            info = Text("  " + "  ".join(parts), style=_STL_MUTED)
            body.append(info)

        if self._logs:
            log_lines: list[Text] = []
            for msg, lvl in list(self._logs)[-8:]:
                style = _LOG_STYLES.get(lvl, _STL_MUTED)
                prefixes = {"active": "●", "done": "✔", "warn": "⚠", "tool": "⚙", "llm": "›", "thought": "🧠", "data": "·", "info": "·"}
                prefix = prefixes.get(lvl, "·")
                log_lines.append(Text(f"  {prefix} {msg}", style=style))
            if log_lines:
                log_panel = Panel(
                    Group(*log_lines),
                    title="Log",
                    border_style="grey46",
                    padding=(0, 1),
                )
                body.append(Text(""))
                body.append(log_panel)

        return Panel(
            Group(*body),
            border_style="blue",
            title="Research Progress",
        )


async def mock_driver() -> None:
    console = Console()
    pg = ResearchProgress(console)
    pg.start()

    phases = [
        ("Query analysis", 1.5, "", ""),
        ("Sub-question planning", 1.2, "", ""),
        ("Sub-Q: What are the key features of agentic AI?", 3.0, "3 clm"),
        ("Sub-Q: Who are the leading contributors?", 2.8, "5 clm"),
        ("Deduplication", 0.5, "rem 2"),
        ("Contradiction detection", 2.0, "1 fnd"),
        ("Report synthesis", 4.0, "8 clm"),
    ]

    pg.log("🧠 Research session started", "thought")

    for name, duration, detail in phases:
        pg.set_tool(f"Processing")
        pg.set_action(name)
        pg.log(f"● {name}", "active")
        pg.phase(name)

        steps = max(1, int(duration / 0.2))
        for _ in range(steps):
            await asyncio.sleep(0.2)
            pg.add_tokens(12)

        pg.add_sources(int(duration * 0.5))
        pg.add_claims(int(duration * 2))
        pg.mark_done(name, detail)
        pg.log(f"✔ {name} — done", "done")

    pg.log("✔ Research complete", "done")
    pg.stop()

    console.print("\n[bold green]Mock dashboard finished.[/bold green]")


if __name__ == "__main__":
    asyncio.run(mock_driver())
