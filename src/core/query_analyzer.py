from __future__ import annotations

from src.llm.llm_client import LLMClient
from src.models import ClarificationResponse, QueryAnalysis
from src.prompts import ANALYZE_SYSTEM_PROMPT, CLARIFY_SYSTEM_PROMPT


class QueryAnalyzer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(self, query: str) -> QueryAnalysis:
        return await self._llm.generate(
            system_prompt=ANALYZE_SYSTEM_PROMPT,
            user_prompt=f"Analyze this research question:\n\n{query}",
            response_model=QueryAnalysis,
            temperature=0.2,
        )

    async def generate_clarification(
        self, query: str, ambiguities: list[str]
    ) -> ClarificationResponse:
        user_prompt = (
            f"Original query: {query}\n\n"
            f"Identified ambiguities:\n"
            + "\n".join(f"- {a}" for a in ambiguities)
        )
        return await self._llm.generate(
            system_prompt=CLARIFY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ClarificationResponse,
            temperature=0.3,
        )
