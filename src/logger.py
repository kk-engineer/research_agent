from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_tracebacks
from rich.console import Group
from rich import box

from src.config import settings
from src.rich_console import console, Theme, make_panel, make_label_value, syntax_block
from src.tracing import tracer
from src.metrics import metrics
from src.telemetry import get_timings

_initialised = False

_step_start: Optional[float] = None
_step_name: str = ""

_JSON_LOG: list[dict[str, Any]] = []


def begin_step(name: str) -> None:
    global _step_start, _step_name
    _step_start = time.monotonic()
    _step_name = name
    logging.getLogger(__name__).info("▸ %s", name)
    tracer.event(name, "general")


def end_step(status: str = "done") -> float:
    global _step_start, _step_name
    if _step_start is None:
        return 0.0
    elapsed = time.monotonic() - _step_start
    log = logging.getLogger(__name__)

    if status == "done":
        log.info("  ✔ %s  [%s]  (%.2fs)", "done", _step_name, elapsed)
    elif status == "timeout":
        log.warning("  ⏱ %s  [%s]  (%.2fs)", "timeout", _step_name, elapsed)
    else:
        log.warning("  ✘ %s  [%s]  (%.2fs)", status, _step_name, elapsed)

    _step_start = None
    _step_name = ""
    return elapsed


@asynccontextmanager
async def trace_step(name: str, logger: logging.Logger) -> AsyncIterator[None]:
    begin_step(name)
    try:
        yield
    except Exception:
        end_step("failed")
        raise
    else:
        end_step("done")


class AgentLogger:
    @staticmethod
    def thought(step: str, detail: str = "") -> str:
        return f"🧠 {step} — {detail}" if detail else f"🧠 {step}"

    @staticmethod
    def action(tool: str, input_preview: str = "") -> str:
        return f"🔧 [{tool}] {input_preview}" if input_preview else f"🔧 [{tool}]"

    @staticmethod
    def llm_call(purpose: str, model: str = "") -> str:
        tag = f"({model})" if model else ""
        return f"🤖 LLM {tag}: {purpose}"

    @staticmethod
    def llm_response(purpose: str, chars: int = 0) -> str:
        return f"  LLM ← {chars} chars  [{purpose}]"

    @staticmethod
    def search_query(provider: str, query: str, n: int = 0) -> str:
        return f"🔍 [{provider}] \"{query}\"  → {n} results"

    @staticmethod
    def page_fetch(url: str, chars: int = 0, status: str = "ok") -> str:
        detail = f" ({chars} chars)" if chars else ""
        return f"🌐 {status}: {url}{detail}"

    @staticmethod
    def extract(url: str, claims: int = 0) -> str:
        return f"📄 {claims} claims from {url}" if claims else f"📄 No claims from {url}"

    @staticmethod
    def deduplicate(before: int, after: int) -> str:
        return f"🧹 Dedup: {before} → {after} ({before - after} removed)"

    @staticmethod
    def contradiction(count: int) -> str:
        return f"⚡ {count} contradiction(s) detected"

    @staticmethod
    def report(claims: int, sources: int, duration: float) -> str:
        return f"📝 Report: {claims} claims, {sources} sources, {duration:.1f}s"

    @staticmethod
    def agent_state(state: str, detail: str = "") -> str:
        return f"🔄 Agent State: {state}" + (f" — {detail}" if detail else "")


def log_agent_state(state: str, detail: str = "") -> None:
    if not settings.show_agent_state:
        return
    msg = AgentLogger.agent_state(state, detail)
    logging.getLogger(__name__).info(msg)


def log_intermediate_step(label: str, content: str = "") -> None:
    if not settings.show_intermediate_steps:
        return
    logging.getLogger(__name__).info("Step: %s  |  %s", label, content[:200] if content else "")


def log_json_event(event_type: str, data: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": settings.session_id,
        "event": event_type,
        "data": data,
    }
    _JSON_LOG.append(record)
    logging.getLogger(__name__).debug(
        "JSON[%s]: %s", event_type, json.dumps(data, default=str)
    )


def export_json_log() -> list[dict[str, Any]]:
    return list(_JSON_LOG)


class FinalSummaryBuilder:
    def __init__(self) -> None:
        self._items: list[tuple[str, str]] = []

    def add(self, label: str, value: str) -> None:
        self._items.append((label, value))

    def render(self) -> Panel:
        total_llm = metrics.llm_call_count
        total_search = metrics.search_count
        elapsed = metrics.total_runtime
        m, s = divmod(int(elapsed), 60)

        table = Table.grid(padding=(0, 2))
        table.add_column(width=24, style=f"bold {Theme.LABEL}")
        table.add_column(style=Theme.INFO)

        for label, value in self._items:
            table.add_row(label, value)

        table.add_row("")
        table.add_row("Runtime", f"{m:02d}:{s:02d}")
        table.add_row("LLM calls", str(total_llm))
        table.add_row("Search calls", str(total_search))
        table.add_row("Total tokens", f"{metrics.total_tokens:,}")
        table.add_row("Total cost", f"${metrics.total_cost:.6f}")
        table.add_row("Slowest op", metrics.slowest_operation())

        return Panel(
            table,
            title="[bold green]Research Complete[/bold green]",
            border_style="green",
            padding=(1, 2),
            box=box.ROUNDED,
        )


fatal_error_panel = None


def log_error(
    step_name: str,
    error: Exception,
    request_payload: Optional[dict[str, Any]] = None,
) -> None:
    global fatal_error_panel
    import traceback

    tb = traceback.format_exc()
    log = logging.getLogger(__name__)
    log.error("Error in %s: %s\n%s", step_name, error, tb)

    content_parts = [
        Text(f"Step: {step_name}", style=f"bold {Theme.ERROR}"),
        Text(f"Error: {error}", style=Theme.ERROR),
    ]
    if request_payload:
        payload_str = json.dumps(request_payload, indent=2, default=str)
        content_parts.append(Text(""))
        content_parts.append(syntax_block(payload_str, "json"))
    content_parts.append(Text(""))
    content_parts.append(Text(tb, style="dim"))

    fatal_error_panel = make_panel(
        Group(*content_parts),
        title=f"[bold {Theme.ERROR}]✘ Fatal Error: {step_name}[/bold {Theme.ERROR}]",
        border_style=Theme.PANEL_ERROR,
    )
    console.print(fatal_error_panel)


def display_final_summary() -> None:
    elapsed = metrics.total_runtime
    m, s = divmod(int(elapsed), 60)

    console.print()
    console.print(metrics.render_summary_panel())

    if settings.show_intermediate_steps or settings.verbosity_level >= 1:
        llm_table = Table.grid(padding=(0, 2))
        llm_table.add_column(width=4)
        llm_table.add_column(width=22)
        llm_table.add_column(width=12)
        llm_table.add_column(width=12)
        llm_table.add_column(width=10)
        llm_table.add_column(width=10)

        llm_table.add_row(
            Text("#", style="bold yellow"),
            Text("Purpose", style="bold yellow"),
            Text("Time", style="bold yellow"),
            Text("Tokens", style="bold yellow"),
            Text("Cost", style="bold yellow"),
            Text("Status", style="bold yellow"),
        )
        for i, c in enumerate(metrics.llm_calls, 1):
            cost_str = f"${c.estimated_cost:.6f}" if c.estimated_cost else "$0"
            status = "✔" if c.success else "✘"
            llm_table.add_row(
                str(i),
                c.purpose[:22],
                f"{c.latency:.2f}s",
                f"{c.total_tokens:,}",
                cost_str,
                status,
            )
        if metrics.llm_calls:
            console.print(
                Panel(
                    llm_table,
                    title="[bold]LLM Call Details[/bold]",
                    border_style="cyan",
                )
            )

    console.print(
        Panel(
            f"Research complete: {m:02d}:{s:02d}  "
            f"|  {metrics.llm_call_count} LLM calls  "
            f"|  {metrics.search_count} searches  "
            f"|  {metrics.total_tokens:,} tokens  "
            f"|  ${metrics.total_cost:.6f}",
            border_style="green",
        )
    )


def _verbosity_to_log_level(verbosity: int) -> int:
    if verbosity >= 3:
        return logging.DEBUG
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    return logging.WARNING


def setup_logging() -> None:
    global _initialised
    if _initialised:
        return
    _initialised = True

    install_rich_tracebacks(show_locals=False)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.verbosity_level <= 1:
        level = max(level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-5s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
