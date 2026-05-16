from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from src.ui.events import EventType, EventBus, event_bus as _global_bus

_PHASE_TO_STAGE: dict[str, str] = {
    "Query analysis": "Query Analysis",
    "Query clarification": "Clarification",
    "Sub-question planning": "Sub-Question Planning",
    "Deduplication": "Processing",
    "Contradiction detection": "Processing",
    "Report synthesis": "Synthesis",
}


def _phase_to_stage(name: str) -> str:
    for prefix, stage in _PHASE_TO_STAGE.items():
        if name == prefix or name.startswith(prefix):
            return stage
    if name.startswith("Sub-Q:") or name.startswith("Sub-Q "):
        return "Web Search"
    return name


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
_STL_SEARCH_QUERY = Style(bold=True, color="dark_orange")
_STL_SEARCH_RESULT = Style(color="green")
_STL_LLM_RESP = Style(bold=True, color="bright_cyan")
_STL_SECTION = Style(bold=True, color="grey58")

_ICONS = {
    _RUNNING: ("\u25cf", _STL_ACTIVE),
    _DONE: ("\u2714", _STL_SUCCESS),
    _FAILED: ("\u2718", _STL_FAIL),
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
    "llm_response": _STL_LLM_RESP,
    "search_query": _STL_SEARCH_QUERY,
    "search_result": _STL_SEARCH_RESULT,
    "data": Style(color="grey74"),
}

_LOG_PREFIXES = {
    "active": "\u25cf", "done": "\u2714", "warn": "\u26a0", "tool": "\u2699",
    "llm": "\u203a", "llm_response": "",
    "search_query": "", "search_result": "",
    "thought": "", "data": "\u00b7", "info": "\u00b7",
}


class ResearchProgress:
    _SPINNER_CHARS = ['\u280b', '\u2819', '\u2839', '\u2838', '\u283c', '\u2834', '\u2826', '\u2827', '\u280f', '\u280f']

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
        self._streaming_text: str = ""

    def start(self) -> None:
        if self._use_live:
            self._live = Live(
                console=self._console,
                refresh_per_second=10,
                get_renderable=self._render,
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
                f"\u23f1 {m:02d}:{s:02d}  "
                f"\U0001f4ca {self._claims_found} claims  "
                f"\U0001f517 {self._sources_found} sources",
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

    def add_stream_token(self, token: str) -> None:
        self._streaming_text += token
        self._refresh()

    def clear_stream(self) -> None:
        self._streaming_text = ""
        self._refresh()

    def set_thought(self, text: str) -> None:
        pass

    def _refresh(self) -> None:
        if self._live:
            self._live.refresh()

    def _completed_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _DONE)

    def _failed_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _FAILED)

    def _active_count(self) -> int:
        return sum(1 for p in self._phases if p.status == _RUNNING)

    @property
    def _spinner_char(self) -> str:
        t = time.monotonic()
        idx = int(t * 10) % len(self._SPINNER_CHARS)
        return self._SPINNER_CHARS[idx]

    def _elapsed_str(self) -> str:
        t = time.monotonic() - self._start
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    def _step_elapsed(self) -> str:
        if self._step_timer == 0:
            return ""
        t = time.monotonic() - self._step_timer
        return f"{t:.1f}s"

    def _render(self):
        step_time = self._step_elapsed()
        table = Table.grid(padding=(0, 2))
        table.add_column(width=6)
        table.add_column(ratio=1)
        table.add_column(width=12)
        table.add_column(width=12)

        for p in self._phases:
            icon, style = _ICONS.get(p.status, ("\u25cf", _STL_MUTED))
            label = _STATUS_LABELS.get(p.status, "?  ")
            dur = f"[{p.duration:.1f}s]" if p.duration is not None else "..."
            name_st = style if p.status in (_DONE, _FAILED) else _STL_ACTIVE
            table.add_row(
                Text(f"{icon} {label}", style=style),
                Text(p.name, style=name_st),
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
        table.add_row("", Text("\u2500\u2500\u2500", style=_STL_HEADER), "", "")
        table.add_row("", Text(summary, style=_STL_HEADER), "", "")

        body: list = [table]
        if self._current_tool or self._action_text:
            parts = []
            if self._current_tool:
                parts.append(f"\u2699 {self._current_tool}")
            if self._action_text:
                dots = "." * (int(time.monotonic() * 2) % 4)
                parts.append(f"{self._spinner_char} {self._action_text}{dots}")
            if step_time:
                parts.append(f"[{step_time}]")
            body.append(Text("  " + "  ".join(parts), style=_STL_MUTED))

        if self._streaming_text:
            body.append(Text("\u2500\u2500 LLM Response \u2500\u2500", style=_STL_SECTION))
            body.append(Text(self._streaming_text, style=_STL_ACTIVE))

        all_logs = list(self._logs)
        search_logs = [(m, l) for m, l in all_logs if l in ("search_query", "search_result")]
        general_logs = [(m, l) for m, l in all_logs if l not in ("search_query", "search_result")]

        if search_logs:
            body.append(Text(""))
            body.append(Text("\u2500\u2500 Search \u2500\u2500", style=_STL_SECTION))
            for msg, lvl in search_logs[-50:]:
                prefix = _LOG_PREFIXES.get(lvl, "\u00b7")
                body.append(Text(f"  {prefix} {msg}", style=_LOG_STYLES.get(lvl, _STL_MUTED)))

        if general_logs:
            body.append(Text(""))
            body.append(Text("\u2500\u2500 Log \u2500\u2500", style=_STL_SECTION))
            for msg, lvl in general_logs[-150:]:
                prefix = _LOG_PREFIXES.get(lvl, "\u00b7")
                body.append(Text(f"  {prefix} {msg}", style=_LOG_STYLES.get(lvl, _STL_MUTED)))

        return Panel(
            Group(*body),
            border_style="yellow",
            title="Research Progress",
        )


@dataclass
class PhaseRecord:
    name: str
    status: str
    duration: Optional[float] = None
    detail: str = ""


class EventProgress:
    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus or _global_bus
        self._start_time: float = 0.0
        self._last_tool: str = ""
        self._last_action: str = ""

    def _emit(self, event_type: EventType, **data: Any) -> None:
        self._bus.emit_kv(event_type, **data)

    # ── Lifecycle ──

    def start(self) -> None:
        self._start_time = time.monotonic()

    def stop(self) -> None:
        self._emit(EventType.AGENT_COMPLETED)
        self._emit(EventType.METRICS, runtime=time.monotonic() - self._start_time)

    # ── Execution graph stages ──

    def set_stage(self, name: str, status: str = "RUNNING", duration: str = "") -> None:
        """Explicitly set execution graph stage status."""
        if status == "RUNNING":
            self._emit(EventType.STAGE_STARTED, stage=name)
        elif status == "COMPLETED":
            self._emit(EventType.STAGE_COMPLETED, stage=name, duration=duration)
        elif status == "FAILED":
            self._emit(EventType.STAGE_FAILED, stage=name, detail=duration)
        elif status == "SKIPPED":
            self._emit(EventType.STAGE_SKIPPED, stage=name)
        elif status == "running":
            self._emit(EventType.STAGE_STARTED, stage=name)
        elif status == "done":
            self._emit(EventType.STAGE_COMPLETED, stage=name, duration=duration)

    def set_stage_progress(self, name: str, progress: str) -> None:
        self._emit(EventType.STAGE_PROGRESS, stage=name, progress=progress)

    # ── Phase management (backward compat + stage mapping) ──

    def phase(self, name: str, status: str = "running", detail: str = "") -> None:
        stage = _phase_to_stage(name)
        if status == "running":
            self._emit(EventType.TASK_STARTED, name=name, detail=detail, stage=stage)
            self._emit(EventType.STAGE_STARTED, stage=stage)
            self._last_action = name
        elif status == "done":
            self._emit(EventType.TASK_COMPLETED, name=name, detail=detail, stage=stage)
            self._emit(EventType.STAGE_COMPLETED, stage=stage, duration=detail)
        elif status == "failed":
            self._emit(EventType.TASK_FAILED, name=name, detail=detail, stage=stage)
            self._emit(EventType.STAGE_FAILED, stage=stage, detail=detail)
        self._emit(EventType.CURRENT_STEP, stage=stage, action=name, provider="", query="", elapsed=0.0)

    def update_phase(self, name: str, status: str, detail: str = "") -> None:
        sm = {"running": "TASK_STARTED", "done": "TASK_COMPLETED", "failed": "TASK_FAILED"}
        et_name = sm.get(status)
        if et_name:
            self._emit(EventType.TASK_UPDATED, name=name, status=status.upper(), detail=detail)
        stage = _phase_to_stage(name)
        if status == "done":
            self._emit(EventType.STAGE_COMPLETED, stage=stage, duration=detail)

    def mark_done(self, name: str, detail: str = "") -> None:
        stage = _phase_to_stage(name)
        self._emit(EventType.TASK_COMPLETED, name=name, detail=detail, stage=stage)
        self._emit(EventType.STAGE_COMPLETED, stage=stage, duration=detail)
        self._emit(EventType.CURRENT_STEP, stage=stage, action=f"{name} done", provider="", query="", elapsed=0.0)

    def mark_failed(self, name: str, detail: str = "") -> None:
        stage = _phase_to_stage(name)
        self._emit(EventType.TASK_FAILED, name=name, detail=detail, stage=stage)
        self._emit(EventType.STAGE_FAILED, stage=stage, detail=detail)

    # ── Current step ──

    def set_current_step(
        self,
        stage: str = "",
        action: str = "",
        provider: str = "",
        query: str = "",
        elapsed: float = 0.0,
    ) -> None:
        self._emit(EventType.CURRENT_STEP, stage=stage, action=action, provider=provider, query=query, elapsed=elapsed)

    def set_action(self, text: str) -> None:
        self._last_action = text
        stage = _phase_to_stage(text)
        self._emit(EventType.CURRENT_STEP, stage=stage, action=text, provider=self._last_tool, query="", elapsed=0.0)
        self._emit(EventType.TIMELINE, message=text, level="active")

    def set_tool(self, tool: str) -> None:
        self._last_tool = tool
        self._emit(EventType.CURRENT_STEP, stage="", action=self._last_action, provider=tool, query="", elapsed=0.0)

    def set_thought(self, text: str) -> None:
        self._emit(EventType.THOUGHT, text=text)

    # ── Tasks ──

    def emit_task(self, name: str, status: str = "WAITING", detail: str = "", provider: str = "") -> None:
        self._emit(EventType.TASK_UPDATED, name=name, status=status, detail=detail, provider=provider)

    # ── Timeline ──

    def emit_timeline(self, message: str, level: str = "info") -> None:
        self._emit(EventType.TIMELINE, message=message, level=level)

    def log(self, message: str, level: str = "info") -> None:
        self._emit(EventType.TIMELINE, message=message, level=level)

    # ── Search results pane ──

    def set_search_results(
        self,
        tavily: int = 0,
        ddg: int = 0,
        unique_urls: int = 0,
        deduped: int = 0,
        failed: int = 0,
        top_domains: list[str] | None = None,
        result_titles: list[str] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "tavily": tavily,
            "ddg": ddg,
            "unique_urls": unique_urls,
            "deduped": deduped,
            "failed": failed,
        }
        if top_domains:
            data["top_domains"] = top_domains
        if result_titles:
            data["result_titles"] = result_titles
        self._emit(EventType.SEARCH_RESULTS, **data)

    # ── Synthesis streaming ──

    def add_stream_token(self, token: str) -> None:
        self._emit(EventType.SYNTHESIS_BUFFER, token=token)

    def clear_stream(self) -> None:
        self._emit(EventType.SYNTHESIS_CLEARED)

    # ── Metrics ──

    def update_metrics(self, **kwargs: Any) -> None:
        self._emit(EventType.METRICS, **kwargs)

    def add_tokens(self, n: int) -> None:
        self._emit(EventType.METRICS, add_tokens=n)

    def set_tokens(self, n: int) -> None:
        self._emit(EventType.METRICS, tokens=n)

    def add_sources(self, n: int) -> None:
        self._emit(EventType.METRICS, add_sources=n)

    def add_claims(self, n: int) -> None:
        self._emit(EventType.METRICS, add_claims=n)
