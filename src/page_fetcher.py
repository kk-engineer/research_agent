from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from src.config import settings
from src.text_processor import extract_meaningful_text

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

MAX_RESPONSE_BYTES = 256 * 1024

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.page_fetch_timeout, connect=8.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _client


async def fetch_page_text(url: str) -> Optional[str]:
    client = _get_client()
    try:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text[:MAX_RESPONSE_BYTES]
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
