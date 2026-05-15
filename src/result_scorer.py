from __future__ import annotations

import re
from datetime import datetime, timezone

from src.models import SearchResult

PRIMARY_DOMAIN_TLDS: set[str] = {".gov", ".edu", ".org", ".int"}
AUTHORITY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\.gov\.?\w*$"), 1.0),
    (re.compile(r"\.edu\.?\w*$"), 0.95),
    (re.compile(r"\.int\.?\w*$"), 0.9),
    (re.compile(r"\.org\.?\w*$"), 0.7),
    (re.compile(r"wikipedia\.org"), 0.75),
    (re.compile(r"reuters\.com"), 0.85),
    (re.compile(r"bloomberg\.com"), 0.85),
    (re.compile(r"nature\.com"), 0.9),
    (re.compile(r"sciencedirect\.com"), 0.9),
]

_DAYS_DECAY = 365.0


class ResultScorer:
    def score(
        self,
        results: list[SearchResult],
        query: str = "",
    ) -> list[tuple[SearchResult, float]]:
        scored: list[tuple[SearchResult, float]] = []
        query_terms = set(query.lower().split()) if query else set()

        for r in results:
            authority = self._domain_authority(r.url)
            relevance = self._snippet_relevance(r.snippet, query_terms)
            freshness = self._freshness_score(r.published_date)
            combined = authority * 0.4 + relevance * 0.4 + freshness * 0.2
            scored.append((r, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _domain_authority(self, url: str) -> float:
        url_lower = url.lower()
        for pattern, score in AUTHORITY_PATTERNS:
            if pattern.search(url_lower):
                return score
        return 0.5

    def _snippet_relevance(self, snippet: str, query_terms: set[str]) -> float:
        if not snippet or not query_terms:
            return 0.5
        snippet_lower = snippet.lower()
        snippet_words = set(snippet_lower.split())
        if not snippet_words:
            return 0.5
        overlap = len(query_terms & snippet_words)
        return min(1.0, overlap / max(len(query_terms), 1))

    def _freshness_score(self, published_date: datetime | None) -> float:
        if published_date is None:
            return 0.5
        try:
            delta = (datetime.now(timezone.utc) - published_date).days
        except Exception:
            return 0.5
        if delta < 30:
            return 1.0
        if delta < 365:
            return max(0.0, 1.0 - delta / _DAYS_DECAY)
        return 0.1
