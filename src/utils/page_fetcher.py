from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from src.config import settings
from src.utils.text_processor import extract_meaningful_text

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.page_fetch_timeout, connect=settings.page_connect_timeout),
            follow_redirects=True,
            headers={"User-Agent": settings.page_user_agent},
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _client


async def fetch_page_text(url: str) -> Optional[str]:
    client = _get_client()
    try:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text[: settings.max_response_bytes]
        return extract_meaningful_text(content)
    except Exception:
        return None


async def fetch_pages_parallel(urls: list[str]) -> list[Optional[str]]:
    tasks = [fetch_page_text(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if isinstance(r, str) else None  # noqa
        for r in results
    ]
