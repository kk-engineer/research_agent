from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import settings
from src.llm_client import LLMClient
from src.models import SubQuestion
from src.prompts import DECOMPOSE_SYSTEM_PROMPT


class ResearchPlan(BaseModel):
    goal: str = ""
    strategy: str = ""


class SubQueryItem(BaseModel):
    id: str = ""
    purpose: str = ""
    query: str
    expected_evidence: list[str] = Field(default_factory=list)
    priority: str = "medium"


class DecompositionResponse(BaseModel):
    research_plan: ResearchPlan = Field(default_factory=ResearchPlan)
    subqueries: list[SubQueryItem] = Field(default_factory=list)


class SubQueryPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def decompose(self, query: str) -> list[SubQuestion]:
        result = await self._llm.generate(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=f"Decompose this research query into sub-queries:\n\n{query}",
            response_model=DecompositionResponse,
            temperature=0.3,
        )

        sq_items = result.subqueries[: settings.max_sub_questions]
        if len(sq_items) < settings.min_sub_questions:
            sq_texts = self._fallback_split(query)
            return [SubQuestion(text=t) for t in sq_texts]

        return [
            SubQuestion(
                text=item.query,
                purpose=item.purpose,
                expected_evidence=item.expected_evidence,
                priority=item.priority,
            )
            for item in sq_items
        ]

    def _fallback_split(self, query: str) -> list[str]:
        return [f"What are the key facts about {query.strip().lower()}?"]
