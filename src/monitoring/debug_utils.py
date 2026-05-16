from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table

from src.config import settings

_debug_console = Console()


def _should_print() -> bool:
    return settings.verbosity_level >= 2


def print_llm_call(system_prompt: str, user_prompt: str, model: str) -> None:
    if not _should_print():
        return
    from src.monitoring.llm_logger import log_llm_call_start
    log_llm_call_start(model, settings.llm_provider, "debug_utils", system_prompt, user_prompt)


def print_llm_response(response: str, purpose: str, elapsed: float) -> None:
    if not _should_print():
        return
    from src.monitoring.llm_logger import log_llm_call_end
    log_llm_call_end(
        purpose, time.time() - elapsed, response,
        success=True, model=settings.llm_model, provider=settings.llm_provider,
    )


def print_llm_error(purpose: str, error: str) -> None:
    if not _should_print():
        return
    _debug_console.print(
        Panel(
            Text(f"{error}", style="red"),
            title=f"[bold red]LLM Error[/bold red]  [{purpose}]",
            border_style="red",
        )
    )


def print_search_query(provider: str, query: str) -> None:
    pass


def print_search_results(
    provider: str, query: str, results: list, elapsed: float
) -> None:
    pass


def print_step(step_name: str, detail: str = "") -> None:
    if not _should_print():
        return
    text = Text(f"▸ {step_name}", style="bold yellow")
    if detail:
        text.append(f"\n  {detail}", style="white")
    _debug_console.print(
        Panel(text, title="[bold yellow]Step[/bold yellow]", border_style="yellow")
    )


def print_claims_summary(
    sub_question: str,
    claims_count: int,
    url_count: int,
    claims_preview: list[str] | None = None,
) -> None:
    if not _should_print():
        return
    text = Text()
    text.append(f"Sub-question: {sub_question}\n", style="bold")
    text.append(f"Claims extracted: {claims_count}\n", style="green")
    text.append(f"URLs processed: {url_count}\n", style="green")

    if claims_preview and claims_count > 0:
        text.append("\nPreview:\n", style="bold")
        for c in claims_preview[:5]:
            text.append(f"  • {c[:100]}\n", style="white")
        if len(claims_preview) > 5:
            text.append(f"  … and {len(claims_preview) - 5} more\n", style="dim")
    elif claims_count == 0:
        text.append("\n[NO CLAIMS FOUND]\n", style="red bold")
    _debug_console.print(
        Panel(text, title="[bold green]Claims Summary[/bold green]", border_style="green")
    )


def print_report_body(report_text: str) -> None:
    if not _should_print():
        return
    _debug_console.print(
        Panel(
            Syntax(report_text, "markdown", word_wrap=True),
            title="[bold magenta]Generated Report[/bold magenta]",
            border_style="magenta",
        )
    )


def print_debug_info(title: str, content: str, style: str = "cyan") -> None:
    if not _should_print():
        return
    _debug_console.print(
        Panel(
            Syntax(content, "markdown", word_wrap=True),
            title=f"[bold {style}]{title}[/bold {style}]",
            border_style=style,
        )
    )
