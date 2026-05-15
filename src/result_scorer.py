from __future__ import annotations

from datetime import datetime, timezone

from src.config import settings
from src.models import SearchResult


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
            combined = (
                authority * settings.authority_weight
                + relevance * settings.relevance_weight
                + freshness * settings.freshness_weight
            )
            scored.append((r, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _domain_authority(self, url: str) -> float:
        url_lower = url.lower()
        for entry in settings.authority_patterns:
            if entry["domain"] in url_lower:
                return entry["score"]
        return settings.default_domain_authority

    def _snippet_relevance(self, snippet: str, query_terms: set[str]) -> float:
        if not snippet or not query_terms:
            return settings.default_snippet_relevance
        snippet_lower = snippet.lower()
        snippet_words = set(snippet_lower.split())
        if not snippet_words:
            return settings.default_snippet_relevance
        overlap = len(query_terms & snippet_words)
        return min(1.0, overlap / max(len(query_terms), 1))

    def _freshness_score(self, published_date: datetime | None) -> float:
        if published_date is None:
            return settings.default_freshness
        try:
            delta = (datetime.now(timezone.utc) - published_date).days
        except Exception:
            return settings.default_freshness
        if delta < settings.freshness_max_days:
            return 1.0
        if delta < settings.days_decay:
            return max(0.0, 1.0 - delta / settings.days_decay)
        return settings.freshness_min_score
