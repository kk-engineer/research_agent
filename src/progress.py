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

# ── Colours ──────────────────────────────────────────────────
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
        self._live: Optional[Live] = None
        self._use_live = use_live
        self._tokens_used: int = 0
        self._sources_found: int = 0
        self._claims_found: int = 0
        self._action_text: str = "Starting…"
        self._current_tool: str = ""
        self._step_timer: float = 0.0
        self._logs: deque = deque(maxlen=200)

    # ── public API ────────────────────────────────────────────

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
        self._action_text = name
        self._refresh()

    def update_phase(self, name: str, status: str, detail: str = "") -> None:
        for p in reversed(self._phases):
            if p.name == name:
                p.status = status
                p.duration = time.monotonic() - self._start
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

    # ── stats ─────────────────────────────────────────────────

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

    # ── render ────────────────────────────────────────────────

    def _render(self) -> Panel:
        elapsed = self._elapsed_str()
        step_time = self._step_elapsed()

        # status + elapsed + tokens row
        status_icon, status_style = _ICONS.get(
            _RUNNING if self._active_count() else _DONE, ("●", _STL_MUTED)
        )
        status_label = "Researching" if self._active_count() or not self._phases else "Complete"

        top = Table.grid(padding=(0, 2))
        top.add_column(width=2)
        top.add_column(width=14)
        top.add_column(width=18)
        top.add_column(width=16)
        top.add_column(width=16)
        top.add_column(width=16)
        top.add_row(
            Text(status_icon, style=status_style),
            Text(status_label, style=_STL_HEADER),
            Text.assemble(("⏱ ", _STL_STAT_LABEL), (elapsed, _STL_STAT_VAL)),
            Text.assemble(("📊 Tokens ", _STL_STAT_LABEL), (f"{self._tokens_used:,}", _STL_STAT_VAL)),
            Text.assemble(("🔗 Sources ", _STL_STAT_LABEL), (f"{self._sources_found}", _STL_STAT_VAL)),
            Text.assemble(("📄 Claims ", _STL_STAT_LABEL), (f"{self._claims_found}", _STL_STAT_VAL)),
        )

        # current tool + step timer
        tool_bar = Text("")
        if self._current_tool:
            tool_text = self._current_tool
            if len(tool_text) > 48:
                tool_text = tool_text[:45] + "…"
            tool_bar = Text.assemble(
                (" ⚙ ", _STL_STAT_LABEL),
                (f"{tool_text}", _STL_TOOL),
                ("  ", Style()),
                (f"[{step_time}]", _STL_DIM) if step_time else ("", Style()),
            )

        # live action
        action_text = self._action_text
        if len(action_text) > 88:
            action_text = action_text[:85] + "…"
        action = Text.assemble(
            (" ▶ ", _STL_STAT_LABEL),
            (action_text, _STL_ACTIVE),
        )

        # rolling log window
        log_lines: list[Text] = []
        for msg, lvl in self._logs:
            style = _LOG_STYLES.get(lvl, _STL_MUTED)
            prefixes = {"active": "●", "done": "✔", "warn": "⚠", "tool": "⚙", "llm": "›", "thought": "🧠", "data": "·"}
            prefix = prefixes.get(lvl, "·")
            log_lines.append(Text(f"  {prefix} {msg}", style=style))

        log_panel = Panel(
            Group(*log_lines),
            title="Activity Log",
            border_style="grey46",
            padding=(0, 1),
            width=120,
        )

        body = Group(
            Text(""),
            top,
            Text(""),
            tool_bar if self._current_tool else Text(""),
            action,
            Text(""),
            log_panel,
        )
        return Panel(body, border_style="blue", title="Research Agent")


# ══════════════════════════════════════════════════════════════
# Mock driver
# ══════════════════════════════════════════════════════════════

async def mock_driver() -> None:
    console = Console()
    pg = ResearchProgress(console)
    pg.start()

    phases = [
        ("Analyzing query", 1.5),
        ("Planning sub-questions", 1.2),
        ("Searching web for 'agentic AI frameworks'", 3.0),
        ("Reading page 1 of 3", 2.5),
        ("Reading page 2 of 3", 2.8),
        ("Reading page 3 of 3", 3.2),
        ("Checking contradictions", 2.0),
        ("Writing report", 4.0),
    ]

    pg.log("🧠 Research session started", "thought")
    pg.log("Initializing AI Research Agent", "info")

    for i, (name, duration) in enumerate(phases):
        pg.set_tool(f"Phase {i+1}/{len(phases)}")
        pg.set_action(name)
        pg.log(f"● {name}", "active")
        pg.phase(name)

        steps = max(1, int(duration / 0.2))
        for _ in range(steps):
            await asyncio.sleep(0.2)
            pg.add_tokens(12)

        pg.add_sources(int(duration * 0.5))
        pg.add_claims(int(duration * 2))
        pg.mark_done(name)
        pg.log(f"✔ {name} — done", "done")

    pg.log("✔ Research complete", "done")
    pg.stop()

    console.print("\n[bold green]Mock dashboard finished.[/bold green]")


if __name__ == "__main__":
    asyncio.run(mock_driver())
