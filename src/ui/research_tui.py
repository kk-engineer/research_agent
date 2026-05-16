from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Grid
from textual.reactive import reactive
from textual.widget import Widget
from rich.text import Text

from src.ui.events import EventType, EventBus, event_bus as _global_bus

_TL_COLORS = {
    "info": "grey74", "done": "green", "warn": "yellow",
    "active": "bright_blue", "thought": "bright_magenta",
    "llm": "bright_cyan", "search_query": "dark_orange",
    "search_result": "green", "tool": "bright_yellow",
    "data": "grey74", "error": "red",
}

_STAGE_ICONS = {
    "WAITING": ("\u25cb", "grey42"),
    "RUNNING": ("\u25cf", "bright_blue"),
    "COMPLETED": ("\u2714", "green"),
    "FAILED": ("\u2718", "red"),
    "RETRYING": ("\u27f3", "yellow"),
    "SKIPPED": ("\u2014", "grey54"),
}

_TASK_ICONS = {
    "WAITING": ("\u25cb", "grey42"),
    "RUNNING": ("\u25cf", "bright_blue"),
    "DONE": ("\u2714", "green"),
    "FAILED": ("\u2718", "red"),
    "RETRYING": ("\u27f3", "yellow"),
}

_STAGE_ORDER = [
    "Query Analysis", "Clarification", "Sub-Question Planning",
    "Web Search", "Processing", "Synthesis",
]


@dataclass
class StageRecord:
    name: str
    status: str = "WAITING"
    detail: str = ""


@dataclass
class TimelineEntry:
    ts: str
    msg: str
    level: str = "info"


@dataclass
class TaskRecord:
    name: str
    status: str = "WAITING"
    detail: str = ""
    provider: str = ""


@dataclass
class CurrentStepRecord:
    stage: str = ""
    action: str = ""
    provider: str = ""
    query: str = ""
    elapsed: float = 0.0


@dataclass
class SearchCounts:
    tavily: int = 0
    ddg: int = 0
    unique_urls: int = 0
    deduped: int = 0
    failed: int = 0
    top_domains: list[str] = field(default_factory=list)


@dataclass
class MetricsData:
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    searches: int = 0
    llm_calls: int = 0
    errors: int = 0
    retries: int = 0
    search_latency_total: float = 0.0
    search_latency_count: int = 0
    active_tasks: int = 0
    start_time: float = 0.0


class TUIState:
    def __init__(self) -> None:
        self.stages: list[StageRecord] = [StageRecord(s) for s in _STAGE_ORDER]
        self.timeline: deque[TimelineEntry] = deque(maxlen=2000)
        self.tasks: dict[str, TaskRecord] = {}
        self.current_step = CurrentStepRecord()
        self.search = SearchCounts()
        self.synthesis_text: list[str] = []
        self.metrics = MetricsData()
        self.query: str = ""
        self.completed = False

    def feed(self, event_type: EventType, data: dict[str, Any]) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        t = event_type.value

        if t == "AGENT_STARTED":
            self.query = data.get("query", "")
            self.metrics.start_time = time.monotonic()
            self._tl(ts, f"Agent started", "active")

        elif t == "AGENT_COMPLETED":
            self.completed = True
            self._tl(ts, "Research complete", "done")

        elif t == "STAGE_STARTED":
            n = data.get("stage", "")
            self._set_stage(n, "RUNNING")
            self._tl(ts, f"Stage: {n}", "active")

        elif t == "STAGE_COMPLETED":
            n = data.get("stage", "")
            d = data.get("duration", "")
            self._set_stage(n, "COMPLETED", d)
            self._tl(ts, f"\u2714 {n} {d}", "done")

        elif t == "STAGE_FAILED":
            n = data.get("stage", "")
            self._set_stage(n, "FAILED")
            self._tl(ts, f"\u2718 {n}", "error")

        elif t == "STAGE_SKIPPED":
            n = data.get("stage", "")
            self._set_stage(n, "SKIPPED")
            self._tl(ts, f"\u2014 {n}", "data")

        elif t == "STAGE_PROGRESS":
            n = data.get("stage", "")
            p = data.get("progress", "")
            self._set_stage(n, "RUNNING", p)

        elif t == "TIMELINE":
            self._tl(ts, data.get("message", ""), data.get("level", "info"))

        elif t == "TASK_STARTED":
            n = data.get("name", "")
            self.tasks[n] = TaskRecord(name=n, status="RUNNING", detail=data.get("detail", ""), provider=data.get("provider", ""))

        elif t == "TASK_COMPLETED":
            n = data.get("name", "")
            if n in self.tasks:
                self.tasks[n].status = "DONE"
                self.tasks[n].detail = data.get("detail", "")

        elif t == "TASK_FAILED":
            n = data.get("name", "")
            if n in self.tasks:
                self.tasks[n].status = "FAILED"

        elif t == "TASK_RETRYING":
            n = data.get("name", "")
            if n in self.tasks:
                self.tasks[n].status = "RETRYING"

        elif t == "TASK_UPDATED":
            n = data.get("name", "")
            if n not in self.tasks:
                self.tasks[n] = TaskRecord(name=n)
            self.tasks[n].status = data.get("status", self.tasks[n].status)
            if data.get("detail"):
                self.tasks[n].detail = data["detail"]

        elif t == "CURRENT_STEP":
            self.current_step = CurrentStepRecord(
                stage=data.get("stage", ""), action=data.get("action", ""),
                provider=data.get("provider", ""), query=data.get("query", ""),
                elapsed=data.get("elapsed", 0.0),
            )

        elif t == "SEARCH_RESULTS":
            for k in ("tavily", "ddg", "unique_urls", "deduped", "failed"):
                if k in data:
                    setattr(self.search, k, data[k])
            if "top_domains" in data:
                self.search.top_domains = data["top_domains"]

        elif t in ("SYNTHESIS_BUFFER", "LLM_STREAM_TOKEN"):
            self.synthesis_text.append(data.get("token", ""))

        elif t in ("SYNTHESIS_CLEARED", "STREAM_CLEARED"):
            self.synthesis_text.clear()

        elif t == "METRICS":
            m = self.metrics
            for k, v in data.items():
                if hasattr(m, k):
                    setattr(m, k, v)

        elif t == "LOG":
            self._tl(ts, data.get("message", ""), data.get("level", "info"))

        elif t == "ERROR":
            self.metrics.errors += 1
            self._tl(ts, f"Error: {data.get('message', '')}", "error")

        elif t == "THOUGHT":
            txt = data.get("text", "")
            if txt:
                self._tl(ts, txt, "thought")

    def _tl(self, ts: str, msg: str, lvl: str = "info") -> None:
        self.timeline.append(TimelineEntry(ts, msg, lvl))

    def _set_stage(self, name: str, status: str, detail: str = "") -> None:
        for s in self.stages:
            if s.name == name:
                s.status = status
                s.detail = detail or s.detail
                return
        for s in self.stages:
            if name.lower() in s.name.lower() or s.name.lower() in name.lower():
                s.status = status
                s.detail = detail or s.detail
                return


class HeaderPane(Widget):
    DEFAULT_CSS = """
    HeaderPane { border: solid $primary; height: 5; padding: 0 1; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("")


class ExecGraphPane(Widget):
    DEFAULT_CSS = """
    ExecGraphPane { border: solid blue; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  Waiting...", style="grey42")


class TimelinePane(Widget):
    DEFAULT_CSS = """
    TimelinePane { border: solid green; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  Waiting...", style="grey42")


class ActiveTasksPane(Widget):
    DEFAULT_CSS = """
    ActiveTasksPane { border: solid yellow; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  No tasks", style="grey42")


class CurrentStepPane(Widget):
    DEFAULT_CSS = """
    CurrentStepPane { border: solid magenta; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  Waiting...", style="grey42")


class SearchResultsPane(Widget):
    DEFAULT_CSS = """
    SearchResultsPane { border: solid cyan; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  No results yet", style="grey42")


class SynthPane(Widget):
    DEFAULT_CSS = """
    SynthPane { border: solid $secondary; padding: 0 1; overflow-y: auto; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("  Waiting for synthesis...", style="grey42")


class MetricsFooter(Widget):
    DEFAULT_CSS = """
    MetricsFooter { border: solid $accent; padding: 0 1; height: 3; }
    """
    content = reactive("")
    def render(self) -> Text:
        return Text.from_markup(self.content) if self.content else Text("")


class ResearchTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #header { height: 5; }
    #main { height: 1fr; layout: vertical; }
    #row1 { height: 2fr; layout: horizontal; }
    #row2 { height: 1fr; layout: horizontal; }
    #row3 { height: 1fr; layout: horizontal; }
    #exec-graph { width: 1fr; }
    #timeline-pane { width: 2fr; }
    #active-tasks { width: 1fr; }
    #current-step { width: 1fr; }
    #search-results { width: 1fr; }
    #synth-pane { width: 1fr; }
    #footer { height: 3; }
    """

    def __init__(
        self,
        research_task: Callable[[EventBus], Awaitable[None]],
        bus: EventBus | None = None,
    ) -> None:
        super().__init__()
        self._research_task = research_task
        self._bus = bus or _global_bus
        self._state = TUIState()

    def compose(self) -> ComposeResult:
        yield HeaderPane(id="header")
        with Vertical(id="main"):
            with Horizontal(id="row1"):
                yield ExecGraphPane(id="exec-graph")
                yield TimelinePane(id="timeline-pane")
            with Horizontal(id="row2"):
                yield ActiveTasksPane(id="active-tasks")
                yield CurrentStepPane(id="current-step")
            with Horizontal(id="row3"):
                yield SearchResultsPane(id="search-results")
                yield SynthPane(id="synth-pane")
        yield MetricsFooter(id="footer")

    def on_mount(self) -> None:
        self.set_interval(1 / 15, self._render_pass)
        asyncio.create_task(self._run_research())

    async def _run_research(self) -> None:
        try:
            await self._research_task(self._bus)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.exit()

    def _render_pass(self) -> None:
        for event in self._bus.drain():
            self._state.feed(event.type, event.data)
        for event in _global_bus.drain():
            self._state.feed(event.type, event.data)

        self.query_one("#header", HeaderPane).content = self._render_header().markup
        self.query_one("#exec-graph", ExecGraphPane).content = self._render_exec_graph().markup
        self.query_one("#timeline-pane", TimelinePane).content = self._render_timeline().markup
        self.query_one("#active-tasks", ActiveTasksPane).content = self._render_active_tasks().markup
        self.query_one("#current-step", CurrentStepPane).content = self._render_current_step().markup
        self.query_one("#search-results", SearchResultsPane).content = self._render_search_results().markup
        self.query_one("#synth-pane", SynthPane).content = self._render_synthesis().markup
        self.query_one("#footer", MetricsFooter).content = self._render_metrics().markup

    def _render_header(self) -> Text:
        elapsed = time.monotonic() - self._state.metrics.start_time if self._state.metrics.start_time else 0
        m, s = divmod(int(elapsed), 60)
        text = Text()
        text.append(" RESEARCH AGENT\n", style="bold bright_white")
        text.append(f" Query: ", style="bold white")
        text.append(f"{self._state.query[:100]}", style="white")
        text.append(f"  Runtime: {m:02d}:{s:02d}", style="bright_cyan")
        if self._state.completed:
            text.append("  COMPLETED", style="bold green")
        return text

    def _render_exec_graph(self) -> Text:
        text = Text()
        text.append(" Execution Graph\n", style="bold blue underline")
        for s in self._state.stages:
            icon, color = _STAGE_ICONS.get(s.status, ("?", "grey42"))
            text.append(f"  {icon} ", style=f"bold {color}")
            text.append(f"{s.name:<24}", style="white" if s.status == "RUNNING" else "grey58")
            text.append(f"{s.status:<10}", style=color)
            if s.detail:
                text.append(f"{s.detail}", style="green")
            text.append("\n")
        return text

    def _render_timeline(self) -> Text:
        text = Text()
        text.append(" Live Timeline (append-only)\n", style="bold green underline")
        for e in list(self._state.timeline)[-100:]:
            text.append(f"  {e.ts} ", style="bold bright_cyan")
            color = _TL_COLORS.get(e.level, "grey74")
            text.append(f"{e.msg}\n", style=color)
        return text

    def _render_active_tasks(self) -> Text:
        text = Text()
        text.append(" Active Tasks\n", style="bold yellow underline")
        tasks = list(self._state.tasks.values())
        if not tasks:
            text.append("  No active tasks\n", style="grey42")
        else:
            for t in tasks:
                icon, color = _TASK_ICONS.get(t.status, ("\u00b7", "grey74"))
                text.append(f"  {icon} ", style=f"bold {color}")
                text.append(f"{t.status:<8}", style=color)
                text.append(f"  {t.name[:40]}", style="white" if t.status == "RUNNING" else "grey58")
                if t.detail:
                    text.append(f"  {t.detail}", style=color)
                text.append("\n")
        return text

    def _render_current_step(self) -> Text:
        cs = self._state.current_step
        text = Text()
        text.append(" Current Step\n", style="bold magenta underline")
        if cs.stage:
            text.append(f"  Stage:   ", style="bold grey58")
            text.append(f"{cs.stage}\n", style="bright_white bold")
        if cs.action:
            dots = "." * (int(time.monotonic() * 2) % 4)
            text.append(f"  Action:  ", style="bold grey58")
            text.append(f"{cs.action}{dots}\n", style="bright_blue")
        if cs.provider:
            text.append(f"  Tool:    ", style="bold grey58")
            text.append(f"{cs.provider}\n", style="bright_yellow")
        if cs.query:
            text.append(f"  Query:   ", style="bold grey58")
            text.append(f"{cs.query[:60]}\n", style="dark_orange")
        elapsed = cs.elapsed or (time.monotonic() - self._state.metrics.start_time if self._state.metrics.start_time else 0)
        if elapsed:
            text.append(f"  Elapsed: ", style="bold grey58")
            text.append(f"{elapsed:.1f}s\n", style="cyan")
        return text

    def _render_search_results(self) -> Text:
        s = self._state.search
        text = Text()
        text.append(" Search Results\n", style="bold cyan underline")
        text.append(f"  Tavily:       {s.tavily}\n", style="bright_white")
        text.append(f"  DDG:          {s.ddg}\n", style="bright_white")
        text.append(f"  Unique URLs:  {s.unique_urls}\n", style="bright_white")
        text.append(f"  Deduplicated: {s.deduped}\n", style="green")
        if s.failed:
            text.append(f"  Failed:       {s.failed}\n", style="red")
        if s.top_domains:
            text.append(f"  Domains: {', '.join(s.top_domains[:5])}\n", style="blue")
        return text

    def _render_synthesis(self) -> Text:
        text = Text()
        text.append(" Streaming Synthesis\n", style="bold $secondary underline")
        content = "".join(self._state.synthesis_text)
        if content:
            text.append(f"  {content[-2000:]}", style="bright_white")
        else:
            text.append("  Waiting for synthesis...", style="grey42")
        return text

    def _render_metrics(self) -> Text:
        m = self._state.metrics
        elapsed = time.monotonic() - m.start_time if m.start_time else 0
        total_tok = m.tokens_in + m.tokens_out
        avg_lat = (m.search_latency_total / m.search_latency_count) if m.search_latency_count > 0 else 0
        success = 0 if (m.searches + m.llm_calls) == 0 else int((1 - m.errors / max(m.searches + m.llm_calls, 1)) * 100)
        text = Text()
        sep = "  \u2502  "
        items = [
            (f"Runtime", f"{elapsed:.0f}s", "bright_cyan"),
            (f"In", f"{m.tokens_in:,}", "bright_cyan"),
            (f"Out", f"{m.tokens_out:,}", "bright_cyan"),
            (f"Cost", f"${m.cost:.4f}", "bright_yellow"),
            (f"Searches", str(m.searches), "bright_cyan"),
            (f"LLM", str(m.llm_calls), "bright_cyan"),
            (f"Latency", f"{avg_lat:.1f}s", "green"),
            (f"Active", str(m.active_tasks), "green"),
            (f"Errors", str(m.errors), "red" if m.errors else "green"),
            (f"Success", f"{success}%", "green"),
        ]
        for i, (label, val, color) in enumerate(items):
            if i > 0:
                text.append(sep, style="grey42")
            text.append(f" {label}: ", style="grey54")
            text.append(val, style=color)
        return text


async def run_research_in_tui(query: str) -> Any:
    from src.core.orchestrator import ResearchOrchestrator

    report: Any = None
    bus = EventBus()

    async def _run(progress_bus: EventBus) -> None:
        nonlocal report
        from rich.console import Console
        progress_bus.emit_kv(EventType.AGENT_STARTED, query=query)
        console = Console()
        orchestrator = ResearchOrchestrator(console=console)
        ep = EventProgress(bus=progress_bus)
        try:
            report = await orchestrator.research(query, skip_clarification=True, progress=ep)
        finally:
            ep.stop()
            await orchestrator.close()

    from src.ui.progress import EventProgress
    app = ResearchTUI(research_task=_run, bus=bus)
    await app.run_async()
    return report
