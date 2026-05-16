from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.ui.rich_console import console, Theme


_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

_DEFAULT_INPUT_PRICE = 1.00
_DEFAULT_OUTPUT_PRICE = 4.00
_LOCAL_MODELS = {"llamacpp", "mock"}


def _estimate_cost(
    model: str, provider: str, prompt_tokens: int, completion_tokens: int
) -> float:
    if provider in _LOCAL_MODELS:
        return 0.0
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        for key, val in _MODEL_PRICING.items():
            if key in model:
                pricing = val
                break
    if pricing is None:
        pricing = (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE)
    input_price, output_price = pricing
    cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
    return round(cost, 6)


@dataclass
class LLMCallRecord:
    model: str
    provider: str
    purpose: str
    temperature: float = 0.3
    max_tokens: int = 2048
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    retry_count: int = 0
    estimated_cost: float = 0.0
    success: bool = True


@dataclass
class SearchRecord:
    provider: str
    query: str
    depth: str = "advanced"
    max_results: int = 10
    latency: float = 0.0
    result_count: int = 0
    success: bool = True


_CACHE_INFO = {"hits": 0, "misses": 0}


class MetricsCollector:
    def __init__(self) -> None:
        self.llm_calls: list[LLMCallRecord] = []
        self.search_calls: list[SearchRecord] = []
        self.start_time: Optional[float] = None
        self.page_fetches: int = 0
        self.page_fetch_errors: int = 0
        self.claims_extracted: int = 0
        self.claims_deduped: int = 0

    def start(self) -> None:
        self.start_time = time.monotonic()
        self.llm_calls.clear()
        self.search_calls.clear()
        self.page_fetches = 0
        self.page_fetch_errors = 0
        self.claims_extracted = 0
        self.claims_deduped = 0

    @property
    def total_runtime(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost(self) -> float:
        return sum(c.estimated_cost for c in self.llm_calls)

    @property
    def total_llm_latency(self) -> float:
        return sum(c.latency for c in self.llm_calls)

    @property
    def total_search_latency(self) -> float:
        return sum(s.latency for s in self.search_calls)

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def search_count(self) -> int:
        return len(self.search_calls)

    @property
    def failed_llm_calls(self) -> int:
        return sum(1 for c in self.llm_calls if not c.success)

    @property
    def failed_searches(self) -> int:
        return sum(1 for s in self.search_calls if not s.success)

    def record_llm_call(self, record: LLMCallRecord) -> None:
        record.estimated_cost = _estimate_cost(
            record.model, record.provider, record.prompt_tokens, record.completion_tokens
        )
        self.llm_calls.append(record)

    def record_search(self, record: SearchRecord) -> None:
        self.search_calls.append(record)

    def record_cache_hit(self) -> None:
        _CACHE_INFO["hits"] += 1

    def record_cache_miss(self) -> None:
        _CACHE_INFO["misses"] += 1

    def slowest_operation(self) -> str:
        all_ops: list[tuple[str, float]] = []
        for c in self.llm_calls:
            all_ops.append((f"LLM/{c.purpose}", c.latency))
        for s in self.search_calls:
            all_ops.append((f"Search/{s.provider}", s.latency))
        if not all_ops:
            return "N/A"
        all_ops.sort(key=lambda x: x[1], reverse=True)
        return f"{all_ops[0][0]} ({all_ops[0][1]:.2f}s)"

    def render_summary_panel(self) -> Panel:
        elapsed = self.total_runtime
        m, s = divmod(int(elapsed), 60)

        table = Table.grid(padding=(0, 2))
        table.add_column(width=22, style=f"bold {Theme.LABEL}")
        table.add_column(style=Theme.INFO)

        table.add_row("Total runtime", f"{m:02d}:{s:02d}")
        table.add_row("LLM calls", str(self.llm_call_count))
        table.add_row("Failed LLM calls", str(self.failed_llm_calls))
        table.add_row("Search calls", str(self.search_count))
        table.add_row("Failed searches", str(self.failed_searches))
        table.add_row("Total tokens", f"{self.total_tokens:,}")
        table.add_row("Total cost", f"${self.total_cost:.6f}")
        table.add_row("Page fetches", str(self.page_fetches))
        table.add_row("Page fetch errors", str(self.page_fetch_errors))
        table.add_row("Claims extracted", str(self.claims_extracted))
        table.add_row("Claims deduped", str(self.claims_deduped))
        table.add_row("Slowest op", self.slowest_operation())
        table.add_row(
            "Cache",
            f"{_CACHE_INFO['hits']} hits / {_CACHE_INFO['misses']} misses",
        )

        stats_panel = Panel(
            table,
            title="[bold green]Performance Metrics[/bold green]",
            border_style="green",
            padding=(1, 2),
            box=box.ROUNDED,
        )
        return stats_panel


metrics = MetricsCollector()
