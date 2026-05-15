from __future__ import annotations

from rich.box import ROUNDED, HEAVY
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree
from rich.columns import Columns
from rich.layout import Layout
from rich import box

console = Console()


class Theme:
    PRIMARY = "bright_cyan"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "white"
    DIM = "grey58"
    HEADER = "bright_white"
    LLM_CALL = "bright_blue"
    LLM_RESP = "green"
    SEARCH = "bright_magenta"
    TOOL = "bright_yellow"
    THOUGHT = "bright_magenta"
    HIGHLIGHT = "cyan"
    MUTED = "grey46"
    TIME = "bright_cyan"
    LABEL = "bright_white"

    PANEL_LLM = "cyan"
    PANEL_SEARCH = "magenta"
    PANEL_SEARCH_REQ = "yellow"
    PANEL_SEARCH_RESP = "green"
    PANEL_ERROR = "red"
    PANEL_SUCCESS = "green"
    PANEL_INFO = "blue"
    PANEL_WARN = "yellow"
    PANEL_TIMELINE = "bright_cyan"


_STYLES = {
    "llm_call": Style(bold=True, color=Theme.LLM_CALL),
    "llm_resp": Style(bold=True, color=Theme.LLM_RESP),
    "tool": Style(bold=True, color=Theme.TOOL),
    "thought": Style(bold=True, color=Theme.THOUGHT),
    "success": Style(bold=True, color=Theme.SUCCESS),
    "error": Style(bold=True, color=Theme.ERROR),
    "warn": Style(bold=True, color=Theme.WARNING),
    "dim": Style(dim=True, color=Theme.DIM),
    "info": Style(color=Theme.INFO),
    "label": Style(bold=True, color=Theme.LABEL),
    "time": Style(bold=True, color=Theme.TIME),
    "header": Style(bold=True, color=Theme.HEADER),
    "search": Style(color=Theme.SEARCH),
}


def styled(text: str, style_name: str = "info") -> Text:
    s = _STYLES.get(style_name)
    if s:
        return Text(text, style=s)
    return Text(text)


def make_panel(
    content,
    title: str = "",
    border_style: str = "blue",
    subtitle: str = "",
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    return Panel(
        content,
        title=title,
        border_style=border_style,
        subtitle=subtitle,
        padding=padding,
        box=ROUNDED,
    )


def make_label_value(label: str, value: str, sep: str = ": ") -> Text:
    return Text.assemble(
        (label, _STYLES["label"]),
        (sep, Theme.DIM),
        (value, _STYLES["info"]),
    )


def make_key_value_table(items: list[tuple[str, str]], title: str = "", key_style: str = "bold") -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=key_style, width=20)
    table.add_column(style="white")
    for k, v in items:
        table.add_row(k, v)
    return table


def syntax_block(code: str, lang: str = "markdown") -> Syntax:
    return Syntax(code, lang, word_wrap=True, theme="monokai", line_numbers=False)
