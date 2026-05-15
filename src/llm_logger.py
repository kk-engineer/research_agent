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
    vl = settings.verbosity_level
    purpose_tag = _short_purpose(purpose)

    tracer.event(f"LLM call: {purpose_tag} ({model})", "llm")

    if vl >= 2:
        meta = make_key_value_table(
            [
                ("Model", model),
                ("Provider", provider),
                ("Temperature", str(temperature)),
                ("Max tokens", str(max_tokens or settings.llm_max_tokens)),
                ("Purpose", purpose),
            ],
        )

        content = Group(
            meta,
            Text(""),
            Panel(
                syntax_block(system_prompt),
                title="[bold]System Prompt[/bold]",
                border_style="dim",
                box=box.ROUNDED,
            ),
            Panel(
                syntax_block(user_prompt),
                title="[bold]User Prompt[/bold]",
                border_style="dim",
                box=box.ROUNDED,
            ),
        )

        console.print(
            make_panel(
                content,
                title=f"[bold {Theme.PANEL_LLM}]🤖 LLM Call: {purpose_tag}[/bold {Theme.PANEL_LLM}]",
                border_style=Theme.PANEL_LLM,
            )
        )
    elif vl >= 1:
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
    vl = settings.verbosity_level
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

    if not success:
        console.print(
            make_panel(
                Text(response_content or "LLM call failed", style=f"bold {Theme.ERROR}"),
                title=f"[bold {Theme.ERROR}]✘ LLM Error: {purpose_tag}[/bold {Theme.ERROR}] ({elapsed:.2f}s)",
                border_style=Theme.PANEL_ERROR,
            )
        )
        return

    if vl >= 2:
        token_info = f"({total_tokens} tokens, In= {prompt_tokens} tokens, Out= {completion_tokens} tokens)" if total_tokens else ""
        if response_content:
            content = Group(
                syntax_block(response_content),
            )
        else:
            content = Text("(empty)", style="dim")
        console.print(
            make_panel(
                content,
                title=f"[bold {Theme.PANEL_LLM}]LLM Response: {purpose_tag}[/bold {Theme.PANEL_LLM}]  {token_info}",
                border_style=Theme.PANEL_LLM,
            )
        )
    elif vl >= 1:
        token_str = f"  ({total_tokens} tokens)" if total_tokens else ""
        console.print(
            f"  [green]✔[/green] LLM [{purpose_tag}]{token_str}  ({elapsed:.2f}s)"
        )


def _short_purpose(purpose: str) -> str:
    return (
        purpose.replace("Response", "").replace("Analysis", "").replace("Decomposition", "Decompose").strip()
        or purpose
    )
