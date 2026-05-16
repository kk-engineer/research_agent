from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class EventType(str, Enum):
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"

    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    STAGE_FAILED = "STAGE_FAILED"
    STAGE_SKIPPED = "STAGE_SKIPPED"
    STAGE_PROGRESS = "STAGE_PROGRESS"

    TIMELINE = "TIMELINE"

    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_RETRYING = "TASK_RETRYING"
    TASK_UPDATED = "TASK_UPDATED"

    CURRENT_STEP = "CURRENT_STEP"

    SEARCH_RESULTS = "SEARCH_RESULTS"

    SYNTHESIS_BUFFER = "SYNTHESIS_BUFFER"
    SYNTHESIS_CLEARED = "SYNTHESIS_CLEARED"

    METRICS = "METRICS"

    THOUGHT = "THOUGHT"
    LOG = "LOG"
    ERROR = "ERROR"

    LLM_STREAM_TOKEN = "LLM_STREAM_TOKEN"
    STREAM_CLEARED = "STREAM_CLEARED"

    LLM_CALL_STARTED = "LLM_CALL_STARTED"
    LLM_CALL_COMPLETED = "LLM_CALL_COMPLETED"
    SEARCH_STARTED = "SEARCH_STARTED"
    SEARCH_COMPLETED = "SEARCH_COMPLETED"
    EXTRACTION_COMPLETED = "EXTRACTION_COMPLETED"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    REPORT_GENERATED = "REPORT_GENERATED"
    SUBQUERY_GENERATED = "SUBQUERY_GENERATED"
    PAGE_FETCH_STARTED = "PAGE_FETCH_STARTED"
    PAGE_FETCH_COMPLETED = "PAGE_FETCH_COMPLETED"
    METRICS_UPDATE = "METRICS_UPDATE"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, maxsize: int = 10000) -> None:
        self._queue: deque[Event] = deque()
        self._maxsize = maxsize
        self._lock = Lock()

    @property
    def queue(self) -> deque[Event]:
        return self._queue

    def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            if len(self._queue) < self._maxsize:
                self._queue.append(Event(type=event_type, data=data or {}))

    def emit_kv(self, event_type: EventType, **data: Any) -> None:
        self.emit(event_type, data)

    def drain(self) -> list[Event]:
        events: list[Event] = []
        with self._lock:
            while self._queue:
                events.append(self._queue.popleft())
        return events


event_bus = EventBus()
tui_mode: bool = False
