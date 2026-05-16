from __future__ import annotations

import asyncio
import functools
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Optional

from src.ui.rich_console import console


_timings: dict[str, list[float]] = defaultdict(list)


def record_timing(label: str, elapsed: float) -> None:
    _timings[label].append(elapsed)


def get_timings() -> dict[str, list[float]]:
    return dict(_timings)


def get_timing_summary(label: str) -> dict[str, float]:
    vals = _timings.get(label, [])
    if not vals:
        return {"count": 0, "total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "total": sum(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def reset_timings() -> None:
    _timings.clear()


@asynccontextmanager
async def timed_async(label: str) -> AsyncIterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        record_timing(label, elapsed)


@contextmanager
def timed(label: str):
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        record_timing(label, elapsed)


def time_method(func):
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        name = func.__qualname__
        start = time.monotonic()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - start
            record_timing(name, elapsed)

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        name = func.__qualname__
        start = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - start
            record_timing(name, elapsed)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
