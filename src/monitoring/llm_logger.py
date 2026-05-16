from __future__ import annotations

import time
from typing import Optional

from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import box

from src.config import settings
from src.monitoring.async_logger import get_async_logger
from src.monitoring.metrics import metrics, LLMCallRecord
from src.monitoring.tracing import tracer
from src.ui import events as _events
from src.ui.rich_console import console, Theme, make_panel, make_key_value_table


def log_llm_call_start(
    model: str,
    provider: str,
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> float:
    start = time.monotonic()
    purpose_tag = _short_purpose(purpose)

    tracer.event(f"LLM call: {purpose_tag} ({model})", "llm")

    show_full = settings.show_llm_prompts

    _events.event_bus.emit_kv(_events.EventType.LLM_CALL_STARTED, purpose=purpose, model=model)

    meta = make_key_value_table(
        [
            ("Model", model),
            ("Provider", provider),
            ("Temperature", str(temperature)),
            ("Max tokens", str(max_tokens or settings.llm_max_tokens)),
            ("Purpose", purpose),
        ],
        key_style="bold yellow",
    )

    if not _events.tui_mode:
        console.print(
            make_panel(
                meta,
                title=f"🤖 LLM Call: {purpose_tag}",
                border_style=Theme.PANEL_LLM,
            )
        )

        if show_full and system_prompt:
            console.print(
                Panel(
                    Syntax(system_prompt, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]System Prompt[/bold]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

        if show_full and user_prompt:
            console.print(
                Panel(
                    Syntax(user_prompt, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]User Prompt[/bold]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

    return start


def log_llm_call_end(
    purpose: str,
    start: float,
    response_content: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    model: str = "",
    provider: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retry_count: int = 0,
) -> None:
    elapsed = time.monotonic() - start
    purpose_tag = _short_purpose(purpose)

    record = LLMCallRecord(
        model=model,
        provider=provider,
        purpose=purpose_tag,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency=elapsed,
        retry_count=retry_count,
        success=success,
    )
    metrics.record_llm_call(record)

    tracer.complete_event(f"LLM response: {purpose_tag}", "llm", elapsed)

    _events.event_bus.emit_kv(
        _events.EventType.LLM_CALL_COMPLETED,
        purpose=purpose_tag,
        tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency=round(elapsed, 2),
        success=success,
    )

    get_async_logger().emit("llm_call", {
        "model": model,
        "purpose": purpose_tag,
        "duration": round(elapsed, 3),
        "tokens": total_tokens,
        "success": success,
    })

    token_info = (
        f"[bold yellow]Tokens:[/bold yellow] {total_tokens:,}  "
        f"(in: {prompt_tokens:,}, out: {completion_tokens:,})"
    ) if total_tokens else ""

    time_info = f"[bold yellow]Time:[/bold yellow] {elapsed:.2f}s"

    if not success:
        if not _events.tui_mode:
            console.print(
                make_panel(
                    Text(response_content or "LLM call failed", style=f"bold {Theme.ERROR}"),
                    title=(
                        f"[bold {Theme.ERROR}]✘ LLM Error: {purpose_tag}[/bold {Theme.ERROR}]  "
                        f"{time_info}"
                    ),
                    border_style=Theme.PANEL_ERROR,
                )
            )
        return

    show_full = settings.show_llm_prompts

    meta = Text.assemble(
        (f"LLM Response: {purpose_tag}\n", "bold"),
    )
    if time_info:
        meta.append(Text.from_markup(time_info + "\n"))
    if token_info:
        meta.append(Text.from_markup(token_info + "\n"))
    if retry_count:
        meta.append(Text(f"Retries: {retry_count}", style=Theme.WARNING))

    if not _events.tui_mode:
        console.print(
            make_panel(
                meta,
                title=f"✔ LLM Response: {purpose_tag}",
                border_style=Theme.PANEL_LLM,
            )
        )

        if show_full and response_content:
            console.print(
                Panel(
                    Syntax(response_content, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]Response Content[/bold]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )


def _short_purpose(purpose: str) -> str:
    return (
        purpose.replace("Response", "").replace("Analysis", "").replace("Decomposition", "Decompose").strip()
        or purpose
    )
