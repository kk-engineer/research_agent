from __future__ import annotations

import asyncio
import logging
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.config import settings
from src.models import SearchResult
from src.monitoring.logger import AgentLogger
from src.monitoring.search_logger import log_search_start, log_search_end

logger = logging.getLogger(__name__)

# Suppress the misleading rename warning from duckduckgo_search v8.x
# (the "ddgs" package on PyPI is not the same library)
warnings.filterwarnings("ignore", message=".*renamed to `ddgs`.*", category=RuntimeWarning)


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...

    async def close(self) -> None:
        pass


class TavilySearchProvider(WebSearchProvider):
    def __init__(self) -> None:
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    async def close(self) -> None:
        if hasattr(self._client, 'aclose'):
            try:
                await self._client.aclose()
            except Exception:
                pass

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        start = time.monotonic()
        logger.info(AgentLogger.search_query("Tavily", query[:80], 0))
        logger.debug("Full Tavily query: %s", query)

        search_start = log_search_start("Tavily", query, max_results, "advanced")

        response = await self._client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )

        elapsed = time.monotonic() - start
        results: list[SearchResult] = []
        for raw in response.get("results", []):
            published = None
            if raw.get("published_date"):
                try:
                    published = datetime.fromisoformat(raw["published_date"])
                except (ValueError, TypeError):
                    published = None

            results.append(
                SearchResult(
                    url=raw.get("url", ""),
                    title=raw.get("title", ""),
                    snippet=raw.get("content", ""),
                    content=raw.get("content"),
                    published_date=published,
                    source=raw.get("source", "tavily"),
                )
            )

        logger.info(
            "  Tavily returned %d results in %.2fs",
            len(results),
            elapsed,
        )
        for i, r in enumerate(results[:3], 1):
            logger.debug("  [%d] %s  |  %s", i, r.title[:60], r.url)
        if len(results) > 3:
            logger.debug("  … and %d more", len(results) - 3)

        log_search_end("Tavily", query, search_start, results, max_results=max_results, depth="advanced")

        return results


class DuckDuckGoSearchProvider(WebSearchProvider):
    def __init__(self) -> None:
        self._ddgs: Any = None

    async def close(self) -> None:
        ddgs, self._ddgs = self._ddgs, None
        if ddgs is not None:
            try:
                ddgs.__exit__(None, None, None)
            except Exception:
                pass

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        from duckduckgo_search import DDGS

        start = time.monotonic()
        logger.info(AgentLogger.search_query("DuckDuckGo", query[:80], 0))

        search_start = log_search_start("DuckDuckGo", query, max_results, "standard")

        if self._ddgs is None:
            self._ddgs = DDGS()

        loop = asyncio.get_running_loop()
        raw_results: list[dict[str, Any]] = await loop.run_in_executor(
            None,
            lambda: list(self._ddgs.text(query, max_results=max_results)),
        )

        results: list[SearchResult] = []
        for r in raw_results:
            results.append(
                SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    source="duckduckgo",
                )
            )

        elapsed = time.monotonic() - start
        logger.info(
            "  DuckDuckGo returned %d results in %.2fs",
            len(results),
            elapsed,
        )

        log_search_end("DuckDuckGo", query, search_start, results, max_results=max_results, depth="standard")

        return results


def get_search_provider() -> WebSearchProvider:
    if settings.search_provider == "duckduckgo":
        return DuckDuckGoSearchProvider()
    return TavilySearchProvider()
