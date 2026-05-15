from __future__ import annotations

import time
from typing import Optional

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box

from src.config import settings
from src.rich_console import console, Theme, syntax_block, make_panel, make_key_value_table
from src.metrics import metrics, LLMCallRecord
from src.tracing import tracer


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

    if show_full:
        meta = make_key_value_table(
            [
                ("Model", model),
                ("Provider", provider),
                ("Temperature", str(temperature)),
                ("Max tokens", str(max_tokens or settings.llm_max_tokens)),
                ("Purpose", purpose),
            ],
        )

        content_parts = [meta, Text("")]

        if system_prompt:
            content_parts.append(
                Panel(
                    Syntax(system_prompt, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]System Prompt[/bold]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

        if user_prompt:
            content_parts.append(
                Panel(
                    Syntax(user_prompt, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]User Prompt[/bold]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

        console.print(
            make_panel(
                Group(*content_parts),
                title=f"[bold {Theme.PANEL_LLM}]🤖 LLM Call: {purpose_tag}[/bold {Theme.PANEL_LLM}]",
                border_style=Theme.PANEL_LLM,
            )
        )
    else:
        console.print(
            f"  🤖 [bold {Theme.LLM_CALL}]LLM[/bold {Theme.LLM_CALL}] "
            f"[{purpose_tag}]  {model}  temp={temperature}",
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

    token_info = (
        f"[bold]Tokens:[/bold] {total_tokens:,}  "
        f"(in: {prompt_tokens:,}, out: {completion_tokens:,})"
    ) if total_tokens else ""

    time_info = f"[bold]Time:[/bold] {elapsed:.2f}s"

    if not success:
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

    if show_full:
        content_parts = []

        if token_info:
            content_parts.append(Text(token_info, style=Theme.INFO))
        if time_info:
            content_parts.append(Text(time_info, style=Theme.INFO))
        if retry_count:
            content_parts.append(Text(f"Retries: {retry_count}", style=Theme.WARNING))

        if response_content:
            content_parts.append(Text(""))
            content_parts.append(
                Panel(
                    Syntax(response_content, "markdown", word_wrap=True, theme="monokai", line_numbers=False),
                    title="[bold]Response[/bold]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
        else:
            content_parts.append(Text("(empty)", style="dim"))

        console.print(
            make_panel(
                Group(*content_parts),
                title=(
                    f"[bold {Theme.PANEL_LLM}]LLM Response: {purpose_tag}[/bold {Theme.PANEL_LLM}]"
                ),
                border_style=Theme.PANEL_LLM,
            )
        )
    else:
        console.print(
            f"  [green]✔[/green] LLM [{purpose_tag}]  {time_info}  "
            f"{token_info}"
        )


def _short_purpose(purpose: str) -> str:
    return (
        purpose.replace("Response", "").replace("Analysis", "").replace("Decomposition", "Decompose").strip()
        or purpose
    )
