from __future__ import annotations

import logging
from collections import defaultdict

from src.config import settings
from src.llm.llm_client import LLMClient
from src.models import ClaimChunk, ContradictionRecord
from src.prompts import DETECT_SYSTEM_PROMPT
from src.utils.utils import extract_json_from_llm_response, safe_get, with_timeout

logger = logging.getLogger(__name__)


class ContradictionDetector:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def detect(self, claims: list[ClaimChunk]) -> list[ContradictionRecord]:
        if len(claims) < 2:
            return []

        candidates = self._build_candidates(claims)
        if not candidates:
            return []

        contradictions: list[ContradictionRecord] = []
        for a, b in candidates[: settings.max_contradiction_pairs]:
            result = await self._check_pair(a, b)
            if result:
                contradictions.append(result)

        return contradictions

    def _build_candidates(
        self, claims: list[ClaimChunk]
    ) -> list[tuple[ClaimChunk, ClaimChunk]]:
        groups: dict[str, list[ClaimChunk]] = defaultdict(list)
        for c in claims:
            if c.source_url:
                key = c.source_url.split("/")[2] if "://" in c.source_url else c.source_url
            else:
                key = "unknown"
            groups[key].append(c)

        sources = list(groups.keys())
        candidates: list[tuple[ClaimChunk, ClaimChunk]] = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                for ca in groups[sources[i]][: settings.max_claims_per_source]:
                    for cb in groups[sources[j]][: settings.max_claims_per_source]:
                        if self._topic_overlap(ca.text, cb.text):
                            candidates.append((ca, cb))
        return candidates

    def _topic_overlap(self, text_a: str, text_b: str) -> bool:
        words_a = {w.lower() for w in text_a.split() if len(w) > settings.contradiction_min_word_length}
        words_b = {w.lower() for w in text_b.split() if len(w) > settings.contradiction_min_word_length}
        overlap = words_a & words_b
        return len(overlap) >= settings.min_overlap_words

    async def _check_pair(
        self, a: ClaimChunk, b: ClaimChunk
    ) -> ContradictionRecord | None:
        user_prompt = (
            f"Claim A (from {a.source_url}):\n{a.text}\n\n"
            f"Claim B (from {b.source_url}):\n{b.text}\n\n"
            "Do these claims contradict each other? Answer with JSON."
        )

        result_text = await with_timeout(
            self._llm.generate(
                system_prompt=DETECT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=None,
                temperature=settings.contradiction_llm_temperature,
                max_tokens=settings.contradiction_max_tokens,
            ),
            timeout_sec=settings.contradiction_check_timeout,
            label="contradiction_check",
        )

        if not result_text or not isinstance(result_text, str):
            return None

        parsed = extract_json_from_llm_response(result_text)
        contradictions = safe_get(parsed, "contradictions", [])
        if not contradictions:
            return None

        c = contradictions[0]
        return ContradictionRecord(
            topic=safe_get(c, "topic", "Unknown topic"),
            impact=safe_get(c, "impact", ""),
            source_a_claim=safe_get(c, "source_a_claim", a.text),
            source_a=safe_get(c, "source_a", a.source_url),
            source_b_claim=safe_get(c, "source_b_claim", b.text),
            source_b=safe_get(c, "source_b", b.source_url),
            recommended_handling=safe_get(c, "recommended_handling", "present_both"),
        )
