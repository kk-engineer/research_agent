from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from src.config import settings


class AsyncFileLogger:
    """Queue-based async structured logger that writes JSON lines to a file.

    Uses an asyncio.Queue to decouple log emission from file I/O,
    so the main async flow is never blocked by disk writes.

    Alternative to Rich console panels: machine-parseable, non-blocking,
    and preserves full structured data for post-hoc analysis.
    """

    def __init__(self, filepath: str | None = None) -> None:
        self._filepath = filepath or settings.log_file or "research_agent.jsonl"
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5000)
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._writer())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def emit(self, event: str, data: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts": time.time(),
            "event": event,
        }
        record.update(data)
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            logging.getLogger(__name__).warning(
                "AsyncLogger queue full, dropping event: %s", event
            )

    async def _writer(self) -> None:
        dirpath = os.path.dirname(self._filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        while self._running:
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                line = json.dumps(record, default=str)
                with open(self._filepath, "a") as f:
                    f.write(line + "\n")
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logging.getLogger(__name__).error(
                    "AsyncLogger write error: %s", exc
                )


_logger_instance: AsyncFileLogger | None = None


def get_async_logger() -> AsyncFileLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AsyncFileLogger()
    return _logger_instance
