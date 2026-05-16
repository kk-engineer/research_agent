from __future__ import annotations

import logging
from datetime import datetime

from src.config import settings
from src.llm.llm_client import LLMClient
from src.models import ClaimChunk, EvidenceItem, ExtractionResponse, SubQuestion
from src.prompts import BATCH_RELEVANCE_SYSTEM_PROMPT
from src.utils.page_fetcher import fetch_pages_parallel
from src.utils.text_processor import extract_domain, extract_relevant_sentences, split_into_chunks
from src.utils.utils import with_timeout

logger = logging.getLogger(__name__)


class SelectiveExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract(self, url: str, sub_question: SubQuestion) -> list[ClaimChunk]:
        texts = await fetch_pages_parallel([url])
        page_text = texts[0]
        if not page_text:
            return []

        chunks = split_into_chunks(page_text, max_chunk_size=settings.extraction_chunk_size)
        chunks = chunks[: settings.max_chunks_per_page]
        if not chunks:
            return []

        domain = extract_domain(url)
        claims = await self._extract_batch(chunks, sub_question, url, domain)

        if not claims and len(chunks) >= settings.max_chunks_per_page:
            more_chunks = split_into_chunks(page_text, max_chunk_size=settings.extraction_chunk_size)
            extra = more_chunks[
                settings.max_chunks_per_page : settings.max_chunks_per_page + settings.extra_fallback_chunks
            ]
            if extra:
                claims = await self._extract_batch(
                    chunks + extra, sub_question, url, domain
                )

        if not claims:
            claims = self._keyword_fallback(page_text, sub_question, url, domain)

        return claims

    def _keyword_fallback(
        self,
        page_text: str,
        sub_question: SubQuestion,
        url: str,
        domain: str,
    ) -> list[ClaimChunk]:
        sentences = extract_relevant_sentences(page_text, sub_question.text)
        if not sentences:
            return []

        now = datetime.utcnow()
        return [
            ClaimChunk(
                text=s,
                source_url=url,
                source_domain=domain,
                domain_authority=settings.default_domain_authority,
                extracted_at=now,
                sub_question_id=sub_question.id,
            )
            for s in sentences
        ]

    async def _extract_batch(
        self,
        chunks: list[str],
        sub_question: SubQuestion,
        url: str,
        domain: str,
    ) -> list[ClaimChunk]:
        sections_str = "\n\n---\n\n".join(
            f"[Section {i+1}]\n{chunk[:settings.chunk_truncation_length]}" for i, chunk in enumerate(chunks)
        )

        user_prompt = (
            f"Research sub-question: {sub_question.text}\n\n"
            f"Web page sections:\n{sections_str}\n\n"
            "Extract ALL relevant factual claims from the sections above."
        )

        result = await with_timeout(
            self._llm.generate(
                system_prompt=BATCH_RELEVANCE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=ExtractionResponse,
                temperature=settings.extraction_llm_temperature,
                max_tokens=settings.extraction_max_tokens,
            ),
            timeout_sec=settings.llm_request_timeout,
            label=f"extract {url[:40]}",
        )

        if not result or not isinstance(result, ExtractionResponse):
            return []

        now = datetime.utcnow()
        claims: list[ClaimChunk] = []
        for item in result.evidence:
            if isinstance(item, EvidenceItem) and len(item.claim.strip()) > settings.min_claim_length:
                claims.append(
                    ClaimChunk(
                        text=item.claim.strip(),
                        source_url=url,
                        source_domain=domain,
                        domain_authority=item.confidence,
                        extracted_at=now,
                        sub_question_id=sub_question.id,
                    )
                )
        return claims
