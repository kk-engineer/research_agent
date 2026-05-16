from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.ui.events import EventType, EventBus, event_bus as _global_bus

logger = logging.getLogger(__name__)

_STAGE_ORDER = [
    "Query Analysis",
    "Clarification",
    "Sub-Question Planning",
    "Web Search",
    "Processing",
    "Synthesis",
]

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
    "COMPLETED": ("\u2714", "green"),
}

_TL_COLORS = {
    "info": "grey74", "done": "green", "warn": "yellow",
    "active": "bright_blue", "thought": "bright_magenta",
    "llm": "bright_cyan", "search_query": "dark_orange",
    "search_result": "green", "tool": "bright_yellow",
    "data": "grey74", "error": "red",
}


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
    duration: str = ""
    retries: int = 0
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
    result_titles: list[str] = field(default_factory=list)


@dataclass
class LLMCallInfo:
    model: str = ""
    provider: str = ""
    purpose: str = ""
    tokens: int = 0
    latency: float = 0.0
    status: str = ""


@dataclass
class MetricsRecord:
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    searches: int = 0
    llm_calls: int = 0
    errors: int = 0
    retries: int = 0
    search_latency_total: float = 0.0
    search_latency_count: int = 0
    active_tasks: int = 0
    queue: int = 0
    sources: int = 0
    claims: int = 0


class DashboardState:
    def __init__(self, query: str) -> None:
        self.query = query
        self.stages: list[StageRecord] = [StageRecord(s) for s in _STAGE_ORDER]
        self.timeline: deque[TimelineEntry] = deque(maxlen=2000)
        self.tasks: dict[str, TaskRecord] = {}
        self.current_step = CurrentStepRecord()
        self.search = SearchCounts()
        self.llm_call = LLMCallInfo()
        self.synthesis_text: list[str] = []
        self.metrics = MetricsRecord()
        self.start_time = time.monotonic()
        self.completed = False

    def feed(self, event_type: EventType, data: dict[str, Any]) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        t = event_type.value

        if t == "AGENT_STARTED":
            self.query = data.get("query", self.query)
            self.start_time = time.monotonic()
            self._tl(ts, f"Agent started \u2014 {self.query}", "active")

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
            self._tl(ts, f"\u2714 {n}  {d}", "done")

        elif t == "STAGE_FAILED":
            n = data.get("stage", "")
            d = data.get("detail", "")
            self._set_stage(n, "FAILED", d)
            self._tl(ts, f"\u2718 {n}", "error")

        elif t == "STAGE_SKIPPED":
            n = data.get("stage", "")
            self._set_stage(n, "SKIPPED")
            self._tl(ts, f"\u2014 {n} skipped", "data")

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
                self.tasks[n].duration = data.get("duration", "")

        elif t == "TASK_FAILED":
            n = data.get("name", "")
            if n in self.tasks:
                self.tasks[n].status = "FAILED"

        elif t == "TASK_RETRYING":
            n = data.get("name", "")
            if n in self.tasks:
                self.tasks[n].status = "RETRYING"
                self.tasks[n].retries += 1
                self._tl(ts, f"Retry {n}", "warn")

        elif t == "TASK_UPDATED":
            n = data.get("name", "")
            if n not in self.tasks:
                self.tasks[n] = TaskRecord(name=n)
            tsk = self.tasks[n]
            tsk.status = data.get("status", tsk.status)
            if data.get("detail"):
                tsk.detail = data["detail"]
            if data.get("duration"):
                tsk.duration = data["duration"]
            if data.get("provider"):
                tsk.provider = data["provider"]

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
            if "result_titles" in data:
                self.search.result_titles = data["result_titles"]
            self.metrics.searches += 1

        elif t in ("SYNTHESIS_BUFFER", "LLM_STREAM_TOKEN"):
            self.synthesis_text.append(data.get("token", ""))

        elif t in ("SYNTHESIS_CLEARED", "STREAM_CLEARED"):
            self.synthesis_text.clear()

        elif t == "METRICS":
            m = self.metrics
            for k, v in data.items():
                if k.startswith("add_"):
                    base = k[4:]
                    if base == "tokens":
                        m.total_tokens += v
                    elif hasattr(m, base):
                        cur = getattr(m, base)
                        setattr(m, base, cur + v)
                elif k == "tokens":
                    m.total_tokens = v
                elif hasattr(m, k):
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

        elif t == "LLM_CALL_STARTED":
            self.llm_call = LLMCallInfo(
                model=data.get("model", ""),
                provider=data.get("provider", ""),
                purpose=data.get("purpose", ""),
                status="running",
            )
            self._tl(ts, f"LLM: {data.get('purpose', '')}", "llm")

        elif t == "LLM_CALL_COMPLETED":
            self.llm_call.tokens = data.get("tokens", 0)
            self.llm_call.latency = data.get("latency", 0.0)
            self.llm_call.status = "completed"
            self.metrics.tokens_in += data.get("prompt_tokens", 0)
            self.metrics.tokens_out += data.get("completion_tokens", 0)
            self.metrics.llm_calls += 1
            tok = data.get("tokens", 0)
            self.metrics.total_tokens += tok
            self._tl(ts, f"LLM \u2192 {tok} tok ({data.get('latency', 0):.1f}s)", "done")

        elif t == "SEARCH_STARTED":
            q = data.get("query", "")
            self._tl(ts, f"Search: {q[:60]}", "search_query")

        elif t == "SEARCH_COMPLETED":
            n = data.get("results_count", 0)
            lat = data.get("latency", 0.0)
            self.metrics.searches += 1
            self.metrics.search_latency_total += lat
            self.metrics.search_latency_count += 1
            self.search.unique_urls += n
            self._tl(ts, f"Search \u2192 {n} results ({lat:.1f}s)", "search_result")

    def _tl(self, ts: str, msg: str, lvl: str = "info") -> None:
        self.timeline.append(TimelineEntry(ts, msg, lvl))

    def _set_stage(self, name: str, status: str, detail: str = "") -> None:
        for s in self.stages:
            if s.name == name:
                s.status = status
                if detail:
                    s.detail = detail
                return
        for s in self.stages:
            if name.lower() in s.name.lower() or s.name.lower() in name.lower():
                s.status = status
                if detail:
                    s.detail = detail
                return


class ResearchDashboard:
    def __init__(
        self,
        query: str,
        bus: EventBus | None = None,
        console: Console | None = None,
    ) -> None:
        self._query = query
        self._bus = bus or _global_bus
        self._console = console or Console()
        self._state = DashboardState(query)

        self._header_pane = Layout(size=5)
        self._exec_pane = Layout()
        self._timeline_pane = Layout()
        self._tasks_pane = Layout()
        self._step_pane = Layout()
        self._llm_req_pane = Layout()
        self._llm_resp_pane = Layout()
        self._footer_pane = Layout(size=4)

        self._layout = self._build_layout()

        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=15,
            screen=True,
            redirect_stderr=False,
            get_renderable=self._render_wrapper,
        )
        self._running = False

    def _render_wrapper(self) -> Layout:
        self._drain_events()
        self._update_all()
        return self._layout

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            self._header_pane,
            Layout(name="body"),
            self._footer_pane,
        )

        body = layout["body"]
        body.split_column(
            Layout(name="row1", ratio=1),
            Layout(name="row2", ratio=2),
            Layout(name="row3", ratio=1),
        )

        body["row1"].split_row(self._exec_pane, self._timeline_pane)
        body["row2"].split_row(self._llm_req_pane, self._llm_resp_pane)
        body["row3"].split_row(self._tasks_pane, self._step_pane)

        self._update_all()
        return layout

    def start(self) -> None:
        self._running = True
        self._state.start_time = time.monotonic()
        self._live.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._update_all()
            self._live.stop()
        except Exception:
            pass

    def _drain_events(self) -> None:
        try:
            for event in self._bus.drain():
                self._state.feed(event.type, event.data)
        except Exception as exc:
            logger.debug("Dashboard drain error: %s", exc)

    def _update_all(self) -> None:
        try:
            self._header_pane.update(self._render_header())
            self._exec_pane.update(Panel(self._render_exec_graph(), title="Execution Graph", border_style="blue"))
            self._timeline_pane.update(Panel(self._render_timeline(), title="Live Timeline", border_style="green"))
            self._tasks_pane.update(Panel(self._render_active_tasks(), title="Active Tasks", border_style="yellow"))
            self._step_pane.update(Panel(self._render_current_step(), title="Current Step", border_style="magenta"))
            self._llm_req_pane.update(Panel(self._render_llm_request(), title="LLM Request", border_style="cyan"))
            self._llm_resp_pane.update(self._render_llm_response())
            self._footer_pane.update(self._render_footer())
        except Exception as exc:
            logger.debug("Dashboard update error: %s", exc)

    # ── Header ──

    def _render_header(self) -> Panel:
        elapsed = time.monotonic() - self._state.start_time
        m, s = divmod(int(elapsed), 60)
        status = "COMPLETED" if self._state.completed else "RUNNING"
        status_color = "green" if self._state.completed else "yellow"
        text = Text.assemble(
            (" RESEARCH AGENT", "bold bright_white"),
            ("\n", ""),
            (f" Query: ", "bold white"),
            (f"{self._state.query[:120]}", "white"),
            ("\n", ""),
            (f" Runtime: {m:02d}:{s:02d}", "bright_cyan"),
            ("  |  ", "grey42"),
            (f"Status: {status}", status_color),
        )
        return Panel(text, border_style="bright_blue", padding=(0, 1))

    # ── Execution Graph ──

    def _render_exec_graph(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(width=3)
        table.add_column(width=24)
        table.add_column(width=10)
        table.add_column(width=8)
        for s in self._state.stages:
            icon, color = _STAGE_ICONS.get(s.status, ("?", "grey42"))
            table.add_row(
                Text(icon, style=f"bold {color}"),
                Text(s.name, style="white" if s.status == "RUNNING" else "grey58"),
                Text(s.status, style=color),
                Text(s.detail, style="green"),
            )
        return table

    # ── Timeline ──

    def _render_timeline(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(width=10)
        table.add_column(ratio=1)
        entries = list(self._state.timeline)
        if not entries:
            table.add_row("", Text(" Waiting for events...", style="grey42"))
        else:
            for e in entries[-75:]:
                color = _TL_COLORS.get(e.level, "grey74")
                table.add_row(
                    Text(e.ts, style="bold bright_cyan"),
                    Text(e.msg, style=color, no_wrap=True),
                )
        return table

    # ── Active Tasks ──

    def _render_active_tasks(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(width=3)
        table.add_column(width=10)
        table.add_column(ratio=1)
        tasks = list(self._state.tasks.values())
        if not tasks:
            table.add_row("", "", Text("No active tasks", style="grey42"))
        else:
            for t in tasks:
                icon, color = _TASK_ICONS.get(t.status, ("\u00b7", "grey74"))
                table.add_row(
                    Text(icon, style=f"bold {color}"),
                    Text(f"{t.status:<8}", style=color),
                    Text(f"{t.name[:48]}", style="white" if t.status == "RUNNING" else "grey58"),
                )
        return table

    # ── Current Step (embeds Search Results when web search active) ──

    def _render_current_step(self) -> Table:
        cs = self._state.current_step
        s = self._state.search
        table = Table.grid(padding=(0, 2))
        table.add_column(width=18, style="bold grey74")
        table.add_column(ratio=1)

        if cs.stage:
            table.add_row("Stage:", Text(cs.stage, style="bright_white bold"))
        elif not cs.action:
            table.add_row("Stage:", Text("Waiting...", style="grey42"))

        if cs.action:
            dots = "." * (int(time.monotonic() * 2) % 4)
            table.add_row("Action:", Text(f"{cs.action}{dots}", style="bright_blue"))

        if cs.provider:
            table.add_row("Tool:", Text(cs.provider, style="bright_yellow"))

        if cs.query:
            table.add_row("Query:", Text(cs.query[:60], style="dark_orange"))

        elapsed = cs.elapsed if cs.elapsed > 0 else (time.monotonic() - self._state.start_time)
        table.add_row("Elapsed:", Text(f"{elapsed:.1f}s", style="cyan"))

        # Search Results — show whenever data exists
        if s.unique_urls > 0 or s.tavily > 0 or s.ddg > 0:
            table.add_row("", Text("\u2500" * 30, style="grey42"))
            table.add_row("Search Results:", Text("", style="bold cyan"))
            if s.tavily:
                table.add_row("Tavily:", Text(str(s.tavily), style="bright_white"))
            if s.ddg:
                table.add_row("DDG:", Text(str(s.ddg), style="bright_white"))
            table.add_row("Unique URLs:", Text(str(s.unique_urls), style="bright_white"))
            table.add_row("Deduplicated:", Text(str(s.deduped), style="green"))
            if s.failed:
                table.add_row("Failed:", Text(str(s.failed), style="red"))
            if s.top_domains:
                table.add_row("Domains:", Text(", ".join(s.top_domains[:4]), style="cyan"))
            if s.result_titles:
                table.add_row("", Text("", style="grey42"))
                for i, title in enumerate(s.result_titles[-8:], 1):
                    table.add_row(f"  [{i}]", Text(title[:50], style="green", no_wrap=True))

        return table

    # ── LLM Request ──

    def _render_llm_request(self) -> Table:
        llm = self._state.llm_call
        table = Table.grid(padding=(0, 2))
        table.add_column(width=14, style="bold grey74")
        table.add_column(ratio=1)

        if llm.purpose:
            table.add_row("Purpose:", Text(llm.purpose, style="bright_white bold"))
        if llm.model:
            table.add_row("Model:", Text(llm.model, style="bright_cyan"))
        if llm.provider:
            table.add_row("Provider:", Text(llm.provider, style="bright_yellow"))
        if llm.status:
            color = {"running": "yellow", "completed": "green", "failed": "red"}.get(llm.status, "grey")
            table.add_row("Status:", Text(llm.status.upper(), style=f"bold {color}"))
        if llm.tokens:
            table.add_row("Tokens:", Text(f"{llm.tokens:,}", style="cyan"))
        if llm.latency:
            table.add_row("Latency:", Text(f"{llm.latency:.2f}s", style="cyan"))

        # Show last query from current step if available
        if self._state.current_step.query:
            table.add_row("Query:", Text(self._state.current_step.query, style="dark_orange"))

        if not llm.purpose and not llm.model:
            table.add_row("", Text("Waiting for LLM calls...", style="grey42"))

        return table

    # ── LLM Response (full text, no truncation) ──

    def _render_llm_response(self) -> Panel:
        text = "".join(self._state.synthesis_text)
        if not text:
            return Panel(
                Text("Waiting for response...", style="grey42"),
                title="LLM Response",
                border_style="bright_yellow",
                padding=(1, 2),
            )
        return Panel(
            Text(text, style="bright_white", overflow="fold"),
            title="LLM Response",
            border_style="bright_yellow",
            padding=(1, 2),
        )

    # ── Footer (Metrics) ──

    def _render_footer(self) -> Panel:
        m = self._state.metrics
        elapsed = time.monotonic() - self._state.start_time
        avg_lat = (m.search_latency_total / m.search_latency_count) if m.search_latency_count > 0 else 0
        denied = m.searches + m.llm_calls
        success = 0 if denied == 0 else int((1 - m.errors / max(denied, 1)) * 100)
        rt_m, rt_s = divmod(int(elapsed), 60)

        def kv(label: str, val: str, style: str = "white") -> list[Text]:
            return [Text(f" {label}: ", style="grey54"), Text(val, style=style)]

        parts1: list[Text] = []
        for item in [
            ("Runtime", f"{rt_m:02d}:{rt_s:02d}", "bright_cyan"),
            ("In", f"{m.tokens_in:,}", "bright_cyan"),
            ("Out", f"{m.tokens_out:,}", "bright_cyan"),
            ("Total", f"{m.total_tokens:,}", "bright_cyan"),
            ("Cost", f"${m.cost:.4f}", "bright_yellow"),
            ("Searches", str(m.searches), "bright_cyan"),
        ]:
            parts1.extend(kv(*item))

        parts2: list[Text] = [Text("  \u2502", style="grey42")]
        for item in [
            ("LLM", str(m.llm_calls), "bright_cyan"),
            ("Latency", f"{avg_lat:.1f}s", "green"),
            ("Active", str(m.active_tasks), "green"),
            ("Errors", str(m.errors), "red" if m.errors else "green"),
            ("Success", f"{success}%", "green"),
        ]:
            parts2.extend(kv(*item))

        return Panel(Group(Text.assemble(*parts1), Text.assemble(*parts2)), border_style="bright_cyan", padding=(0, 1))
