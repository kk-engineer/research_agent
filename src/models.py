from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    ANSWERED = "answered"
    PARTIAL = "partial"
    UNANSWERABLE = "unanswerable"


class ContradictionResolution(str, Enum):
    PRESENT_BOTH = "present_both"
    PREFER_NEWER = "prefer_newer"
    PREFER_AUTHORITATIVE = "prefer_authoritative"


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    source: str = ""


class QueryAnalysis(BaseModel):
    original_query: str
    primary_intent: str
    research_type: str = ""
    expected_output: str = ""
    user_persona: str = ""
    domain: str = ""
    scope: str
    requires_comparison: bool = False
    requires_ranking: bool = False
    requires_recommendation: bool = False
    evaluation_criteria: list[str] = Field(default_factory=list)
    must_answer_questions: list[str] = Field(default_factory=list)
    search_intents: list[str] = Field(default_factory=list)
    constraints: dict[str, str | None] = Field(default_factory=dict)
    ambiguities: list[str] = Field(default_factory=list)
    has_ambiguities: bool = False


class ClarificationResponse(BaseModel):
    needs_clarification: bool = True
    question: str = ""
    resolved_query: str = ""
    assumptions: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    purpose: str = ""
    expected_evidence: list[str] = Field(default_factory=list)
    priority: str = "medium"
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    claims: list[ClaimChunk] = Field(default_factory=list)
    search_attempts: int = 0


class ClaimChunk(BaseModel):
    claim_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str
    source_url: str
    source_domain: str
    domain_authority: float = 0.5
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    sub_question_id: str = ""
    embedding: Optional[list[float]] = None


class EvidenceItem(BaseModel):
    claim: str
    relevance: str = ""
    evidence_type: str = ""
    confidence: float = 0.5


class ContradictionRecord(BaseModel):
    topic: str
    impact: str = ""
    source_a_claim: str
    source_a: str
    source_b_claim: str
    source_b: str
    recommended_handling: str = "present_both"


class ExtractionResponse(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class SynthesisSection(BaseModel):
    heading: str = ""
    insights: str = ""
    key_evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class SynthesisRecommendation(BaseModel):
    name: str = ""
    why_recommended: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    best_for: str = ""


class SynthesisComparisonTable(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class SynthesisResponse(BaseModel):
    title: str = ""
    direct_answer: str = ""
    executive_summary: str = ""
    recommendations: list[SynthesisRecommendation] = Field(default_factory=list)
    detailed_analysis_sections: list[SynthesisSection] = Field(default_factory=list)
    comparison_table: SynthesisComparisonTable = Field(default_factory=SynthesisComparisonTable)
    tradeoffs_and_limitations: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class CoverageGap(BaseModel):
    sub_question_id: str
    sub_question_text: str
    status: SubQuestionStatus
    note: str = ""


class ReportSection(BaseModel):
    heading: str
    body: str
    citations: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    why_recommended: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    best_for: str = ""


class ComparisonTable(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class Report(BaseModel):
    query: str
    title: str = ""
    direct_answer: str = ""
    executive_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    comparison_table: ComparisonTable = Field(default_factory=ComparisonTable)
    tradeoffs_and_limitations: list[str] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_claims: int = 0
    web_search_calls: int = 0
