from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.config import settings
from src.core.orchestrator import ResearchOrchestrator
from src.llm.llm_client import get_llm_client, LLMClient
from src.monitoring.async_logger import get_async_logger
from src.monitoring.logger import setup_logging, display_final_summary, log_error
from src.monitoring.metrics import metrics
from src.monitoring.tracing import tracer
from src.output.report_formatter import format_report
from src.ui import events as _events
from src.ui.research_tui import run_research_in_tui
from src.ui.dashboard import ResearchDashboard
from src.ui.progress import EventProgress

logger = logging.getLogger(__name__)
console = Console()


def _slugify(text: str, max_len: int = 60) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9\s_-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug[:max_len].rstrip("_")


_REPORTS_DIR = Path("research_reports")


def _default_report_path(query: str) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(query) or "research"
    path = _REPORTS_DIR / f"{slug}_research_report.md"
    counter = 1
    while path.exists():
        path = _REPORTS_DIR / f"{slug}_research_report_{counter}.md"
        counter += 1
    return path


def _save_report(markdown: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    logger.info("Report saved to %s", path)


def _clickable_path(path: Path) -> str:
    return f"file://{path.resolve()}"


def _copy_to_clipboard(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except (ImportError, Exception) as exc:
        logger.debug("Clipboard copy failed: %s", exc)
        return False


def _display_report(markdown: str) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(markdown),
            title="[bold green]Research Report[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _offer_report_actions(markdown: str, query: str) -> None:
    help_text = (
        "[bold]s[/bold] save to file  "
        "[bold]c[/bold] copy to clipboard  "
        "[bold]b[/bold] both  "
        "[Enter] skip"
    )
    choice = Prompt.ask(help_text, choices=["s", "c", "b", ""], default="")
    choice = choice.strip().lower()

    if choice in ("s", "b"):
        path = _default_report_path(query)
        _save_report(markdown, path)
        console.print(f"[green]Report saved:[/green] [link={_clickable_path(path)}]{path}[/link]")

    if choice in ("c", "b"):
        ok = _copy_to_clipboard(markdown)
        if ok:
            console.print("[green]Copied to clipboard[/green]")
        else:
            console.print(
                "[yellow]Clipboard copy unavailable[/yellow]\n"
                "  Install [bold]pyperclip[/bold] or copy manually."
            )


def _banner() -> Panel:
    logo = Table.grid(padding=(0, 1))
    logo.add_row(r"[bold cyan] ___      _        ___       _   _    _ [/bold cyan]")
    logo.add_row(r"[bold cyan]| _ \ ___| |__ ___/ __| __ _| |_| |_ (_)[/bold cyan]")
    logo.add_row(r"[bold cyan]|   // _ \ '_ \/ _ \__ \/ _` |  _| ' \| |[/bold cyan]")
    logo.add_row(r"[bold cyan]|_|_\\___/_.__/\___/___/\__,_|\__|_||_|_|[/bold cyan]")

    line1 = Text.assemble(
        (f"  LLM: {settings.llm_provider}/{settings.llm_model}", "white"),
        ("  |  ", "dim"),
        (f"Search: {settings.search_provider}", "white"),
        ("  |  ", "dim"),
        (f"Verbosity: {settings.verbosity_level}", "white"),
    )
    line2 = Text.assemble(
        (
            f"  Max sub-questions: {settings.min_sub_questions}\u2013{settings.max_sub_questions}",
            "white",
        ),
        ("  |  ", "dim"),
        (f"Max tool calls: {settings.max_tool_calls}", "white"),
        ("  |  ", "dim"),
        (f"Claims: {'OFF' if settings.disable_claims else 'ON'}", "yellow" if settings.disable_claims else "green"),
    )

    return Panel(
        Group(logo, Text(""), line1, line2),
        border_style="cyan",
        subtitle="autonomous research assistant",
    )


def _validate_config() -> None:
    try:
        settings.validate()
        console.print("[green]Configuration is valid.[/green]")
        info = Table.grid(padding=(0, 2))
        info.add_row("LLM provider:", f"[cyan]{settings.llm_provider}[/cyan]")
        info.add_row("LLM model:", f"[cyan]{settings.llm_model}[/cyan]")
        info.add_row("Search provider:", f"[cyan]{settings.search_provider}[/cyan]")
        if settings.llm_provider == "llamacpp":
            info.add_row("Base URL:", f"[cyan]{settings.llama_base_url}[/cyan]")
        info.add_row("Verbosity:", f"[cyan]{settings.verbosity_level}[/cyan]")
        info.add_row("Claims extraction:", f"[cyan]{'disabled' if settings.disable_claims else 'enabled'}[/cyan]")
        console.print(Panel(info, title="Configuration"))
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)


async def run_single(
    query: str,
    *,
    output_path: str = "",
    skip_clarification: bool = False,
    no_save: bool = False,
    copy: bool = False,
    llm_client: Optional[LLMClient] = None,
) -> None:
    tracer.start()
    metrics.start()

    previous_tui = _events.tui_mode
    _events.tui_mode = True

    dashboard = ResearchDashboard(query)
    dashboard.start()

    event_progress = EventProgress()
    event_progress._emit(_events.EventType.AGENT_STARTED, query=query)

    orchestrator = ResearchOrchestrator(llm=llm_client, console=console)
    try:
        report = await orchestrator.research(
            query,
            skip_clarification=skip_clarification,
            progress=event_progress,
        )
    except asyncio.CancelledError:
        dashboard.stop()
        _events.tui_mode = previous_tui
        console.print("\n[yellow]Research cancelled[/yellow]")
        return
    except Exception as e:
        dashboard.stop()
        _events.tui_mode = previous_tui
        log_error("research", e)
        return
    finally:
        await orchestrator.close()

    dashboard.stop()
    _events.tui_mode = previous_tui

    report_doc = format_report(report)
    _display_report(report_doc)
    display_final_summary()

    saved_path: Optional[Path] = None

    if copy:
        ok = _copy_to_clipboard(report_doc)
        console.print(
            "[green]Copied to clipboard[/green]"
            if ok
            else "[yellow]Clipboard copy unavailable[/yellow]"
        )

    if output_path:
        saved_path = Path(output_path)
        _save_report(report_doc, saved_path)
    elif not no_save:
        saved_path = _default_report_path(query)
        _save_report(report_doc, saved_path)

    if saved_path:
        console.print(f"[green]Report saved:[/green] [link={_clickable_path(saved_path)}]{saved_path}[/link]")

    if no_save and not output_path and not copy:
        _offer_report_actions(report_doc, query)


async def run_interactive(llm_client: Optional[LLMClient] = None) -> None:
    welcome = Panel(
        "Ask any research question in natural language.\n\n"
        "[bold]/help[/bold]  Show commands  [bold]/quit[/bold]  Exit\n"
        "[bold]Ctrl+C[/bold] during research cancels and returns to prompt",
        title="[bold cyan]AI Research Assistant[/bold cyan]",
        border_style="cyan",
    )
    console.print(welcome)

    while True:
        try:
            query = console.input("\n[bold cyan]research>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not query:
            continue

        if query.startswith("/"):
            command = query.lower()
            if command in ("/quit", "/exit"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif command == "/help":
                console.print(
                    "Commands:\n"
                    "  /quit, /exit  Exit\n"
                    "  /help         This help message\n"
                    "  /clear        Clear the screen\n"
                    "  Ctrl+C        Cancel current research\n"
                    "\nOtherwise, type your research question."
                )
            elif command == "/clear":
                console.clear()
            else:
                console.print(f"[red]Unknown command:[/red] {query}")
            continue

        tracer.start()
        metrics.start()

        previous_tui = _events.tui_mode
        _events.tui_mode = True

        dashboard = ResearchDashboard(query)
        dashboard.start()

        event_progress = EventProgress()
        event_progress._emit(_events.EventType.AGENT_STARTED, query=query)

        orchestrator = ResearchOrchestrator(llm=llm_client, console=console)
        try:
            report = await orchestrator.research(query, progress=event_progress)
        except asyncio.CancelledError:
            dashboard.stop()
            _events.tui_mode = previous_tui
            console.print("[yellow]Research cancelled \u2014 returning to prompt[/yellow]")
            continue
        except Exception as e:
            dashboard.stop()
            _events.tui_mode = previous_tui
            log_error("research", e)
            continue
        finally:
            await orchestrator.close()

        dashboard.stop()
        _events.tui_mode = previous_tui

        report_doc = format_report(report)
        _display_report(report_doc)
        display_final_summary()
        _offer_report_actions(report_doc, query)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Research Assistant Agent \u2014 autonomous multi-source research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Arguments:\n"
            "  query                    Research query in natural language\n"
            "  --interactive, -i        Run in interactive (REPL) mode\n"
            "  --output, -o   FILE      Save report to FILE (markdown)\n"
            "  --copy, -c               Copy report to clipboard on completion\n"
            "  --no-save                Skip auto-saving report to file\n"
            "  --skip-clarification     Skip asking clarifying questions\n"
            "  --validate-config        Validate configuration and exit\n"
            "  --verbose, -v            Enable debug-level logging (sets verbosity=debug)\n"
            "  --verbose-debug          Print detailed LLM/search/debug panels (sets verbosity=trace)\n"
            "  --verbosity, -V          Set verbosity: minimal | normal | debug | trace\n"
            "\n"
            "Interactive commands (with -i / --interactive):\n"
            "  /help              Show this interactive help\n"
            "  /quit, /exit       Exit the interactive shell\n"
            "  /clear             Clear the screen\n"
            "  Ctrl+C             Cancel current research\n"
            "\n"
            "Examples:\n"
            "  research_agent \"What are the latest AI trends?\"\n"
            "  research_agent -i\n"
            "  research_agent \"X\" --output report.md --skip-clarification\n"
            "  research_agent \"X\" --verbose\n"
            "  research_agent \"X\" --verbosity trace\n"
            "  research_agent \"X\" --tui\n"
            "  research_agent \"X\" --tui --show-prompts\n"
            "\n"
            "Environment variables (see .env):\n"
            "  TAVILY_API_KEY    Search API key (required for tavily)\n"
            "  LLM_API_KEY       OpenAI API key (required for openai provider)\n"
            "\n"
            "Configuration file: config.toml\n"
            "  All non-secret settings including LLM, search, performance, logging, and claims.\n"
            "\n"
        ),
    )
    parser.add_argument("query", type=str, nargs="?", help="Research query in natural language")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--output", "-o", type=str, default="", help="Output file path for the report (markdown)")
    parser.add_argument("--copy", "-c", action="store_true", help="Copy report to clipboard automatically")
    parser.add_argument("--no-save", action="store_true", help="Skip auto-saving report to file")
    parser.add_argument("--skip-clarification", action="store_true", help="Skip asking clarifying questions")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Set verbosity to debug")
    parser.add_argument("--verbose-debug", action="store_true", help="Set verbosity to trace (full prompt/response logging)")
    parser.add_argument("--tui", action="store_true", help="Launch Textual TUI dashboard")
    parser.add_argument("--show-prompts", action="store_true", default=False,
                        help="Show LLM prompts and responses in TUI (default: hidden)")
    parser.add_argument("--verbosity", "-V", type=str,
                        choices=["minimal", "normal", "debug", "trace"],
                        default=None,
                        help="Set verbosity level (overrides config.toml)")

    args = parser.parse_args()

    if args.verbosity is not None:
        vmap = {"minimal": 0, "normal": 1, "debug": 2, "trace": 3}
        settings.verbosity_level = vmap[args.verbosity]
    elif args.verbose_debug:
        settings.verbosity_level = 3
    elif args.verbose:
        settings.verbosity_level = 2

    if settings.verbosity_level >= 2:
        os.environ["LOG_LEVEL"] = "DEBUG"

    setup_logging()
    settings.validate()
    get_async_logger().start()

    if args.validate_config:
        _validate_config()
        return

    if not args.query and not args.interactive:
        parser.print_help()
        console.print("\n[yellow]Provide a query or use --interactive mode.[/yellow]")
        sys.exit(1)

    try:
        settings.validate()
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        console.print("Create a [bold].env[/bold] file with the required API keys.")
        console.print("Create/edit [bold]config.toml[/bold] for non-secret settings.")
        sys.exit(1)

    try:
        llm_client = get_llm_client()
        await llm_client.check_connectivity()
    except ConnectionError as e:
        logger.error("Model connectivity check failed: %s", e)
        console.print(f"[red]Error:[/red] {e}")
        console.print("Make sure your LLM/embedding server is running and the URLs in [bold]config.toml[/bold] are correct.")
        sys.exit(1)

    if args.tui:
        if not args.query:
            console.print("[red]Error: --tui requires a query.[/red]")
            sys.exit(1)
        _events.tui_mode = True
        try:
            report = await run_research_in_tui(args.query)
        finally:
            _events.tui_mode = False
        if report:
            report_doc = format_report(report)
            _display_report(report_doc)
            display_final_summary()
        return

    console.print(_banner())

    try:
        if args.interactive:
            await run_interactive(llm_client=llm_client)
        elif args.query:
            await run_single(
                args.query,
                output_path=args.output,
                skip_clarification=args.skip_clarification,
                no_save=args.no_save,
                copy=args.copy,
                llm_client=llm_client,
            )
    finally:
        await get_async_logger().stop()
        await llm_client.close()


def entry_point() -> None:
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_exit(signum: int, frame: object) -> None:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Research cancelled by user[/yellow]")
        sys.exit(0)
    except asyncio.CancelledError:
        console.print("\n[yellow]Research cancelled[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    entry_point()
