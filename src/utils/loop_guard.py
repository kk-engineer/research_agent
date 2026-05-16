from __future__ import annotations

import hashlib
from typing import Set

from src.config import settings


class LoopGuard:
    def __init__(self) -> None:
        self._executed: Set[str] = set()
        self._tool_call_count: int = 0

    def check_and_register(self, query: str, source: str) -> bool:
        if self._tool_call_count >= settings.max_tool_calls:
            return False

        fp = self._fingerprint(query, source)
        if fp in self._executed:
            return False

        self._executed.add(fp)
        self._tool_call_count += 1
        return True

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    def _fingerprint(self, query: str, source: str) -> str:
        content = f"{query.strip().lower()}|{source.strip().lower()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def reset(self) -> None:
        self._executed.clear()
        self._tool_call_count = 0
