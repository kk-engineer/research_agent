from __future__ import annotations

import logging

from src.config import settings
from src.llm_client import LLMClient
from src.models import (
    ClaimChunk,
    ComparisonTable,
    ContradictionRecord,
    CoverageGap,
    Recommendation,
    Report,
    ReportSection,
    SubQuestion,
    SubQuestionStatus,
    SynthesisResponse,
)
from src.prompts import SYNTHESIS_SYSTEM_PROMPT
from src.utils import with_timeout

logger = logging.getLogger(__name__)


class SynthesisEngine:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        sub_questions: list[SubQuestion],
        contradictions: list[ContradictionRecord],
    ) -> Report:
        all_claims: list[ClaimChunk] = []
        for sq in sub_questions:
            all_claims.extend(sq.claims)

        user_prompt = self._build_prompt(sub_questions, all_claims, contradictions)

        response = await with_timeout(
            self._llm.generate(
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=SynthesisResponse,
                temperature=settings.synthesis_llm_temperature,
                max_tokens=settings.synthesis_max_tokens,
            ),
            timeout_sec=settings.llm_request_timeout,
            label="report_synthesis",
        )

        report = self._parse_result(response, sub_questions)
        report.contradictions = contradictions
        report.total_claims = len(all_claims)

        all_urls = set(c.source_url for c in all_claims)
        report.references = sorted(all_urls)

        return report

    def _build_prompt(
        self,
        sub_questions: list[SubQuestion],
        claims: list[ClaimChunk],
        contradictions: list[ContradictionRecord],
    ) -> str:
        lines: list[str] = []
        lines.append("# Research Data\n")
        lines.append(f"Total claims extracted: {len(claims)}\n")

        for sq in sub_questions:
            lines.append(f"## Sub-Question: {sq.text}")
            lines.append(f"Status: {sq.status.value}")
            if sq.claims:
                for c in sq.claims:
                    lines.append(f"- {c.text}  [{c.source_url}]")
            else:
                lines.append("- (No claims found)")
            lines.append("")

        if contradictions:
            lines.append("## Contradictions Detected")
            for c in contradictions:
                lines.append(f"- Topic: {c.topic}")
                if c.impact:
                    lines.append(f"  Impact: {c.impact}")
                lines.append(f"  Source A: {c.source_a_claim}  [from {c.source_a}]")
                lines.append(f"  Source B: {c.source_b_claim}  [from {c.source_b}]")
            lines.append("")

        return "\n".join(lines)

    def _build_coverage_gaps(
        self, sub_questions: list[SubQuestion]
    ) -> list[CoverageGap]:
        gaps: list[CoverageGap] = []
        for sq in sub_questions:
            if sq.status == SubQuestionStatus.ANSWERED:
                continue
            if not sq.claims:
                gaps.append(
                    CoverageGap(
                        sub_question_id=sq.id,
                        sub_question_text=sq.text,
                        status=SubQuestionStatus.UNANSWERABLE,
                        note="No supporting claims found from any source.",
                    )
                )
            elif sq.status == SubQuestionStatus.PARTIAL:
                gaps.append(
                    CoverageGap(
                        sub_question_id=sq.id,
                        sub_question_text=sq.text,
                        status=SubQuestionStatus.PARTIAL,
                        note="Only partial information was found.",
                    )
                )
        return gaps

    def _build_executive_summary(
        self,
        sections: list[ReportSection],
        sub_questions: list[SubQuestion],
    ) -> str:
        lines: list[str] = []

        total_claims = sum(len(sq.claims) for sq in sub_questions)
        all_urls = {c.source_url for sq in sub_questions for c in sq.claims}
        answered = sum(1 for sq in sub_questions if sq.status == SubQuestionStatus.ANSWERED)

        section_summaries: list[str] = []
        for section in sections:
            heading = section.heading.strip()
            body = section.body.strip()
            if not heading or not body:
                continue
            first_sentence = body.split(".")[0].strip()
            if len(first_sentence) > 200:
                first_sentence = first_sentence[:200] + "..."
            section_summaries.append(f"- **{heading}**: {first_sentence}.")

        if section_summaries:
            lines.append(
                f"This report synthesizes findings from **{len(all_urls)}** sources "
                f"across **{len(sub_questions)}** research sub-questions."
            )
            if answered:
                lines.append(f"**{answered}** sub-question(s) were fully answered.")
            lines.append("")
            lines.append("Key findings:")
            lines.extend(section_summaries)
        else:
            answered_str = ""
            if answered:
                answered_str = f" **{answered}** sub-question(s) were fully answered."
            lines.append(
                f"Research covered **{len(sub_questions)}** sub-question(s) "
                f"using **{total_claims}** claims from **{len(all_urls)}** sources."
                f"{answered_str}"
            )

        return "\n".join(lines)

    def _parse_result(
        self,
        response: SynthesisResponse | None,
        sub_questions: list[SubQuestion],
    ) -> Report:
        if response is None or (
            not response.executive_summary
            and not response.detailed_analysis_sections
            and not response.direct_answer
        ):
            return self._fallback_report(sub_questions)

        sections = [
            ReportSection(
                heading=s.heading,
                body=s.insights,
                citations=s.citations,
            )
            for s in response.detailed_analysis_sections
        ]

        executive_summary = response.executive_summary.strip()
        if not executive_summary and sections:
            executive_summary = self._build_executive_summary(sections, sub_questions)

        if response.recommendations:
            recommendations = [
                Recommendation(
                    name=r.name,
                    why_recommended=r.why_recommended,
                    strengths=r.strengths,
                    weaknesses=r.weaknesses,
                    best_for=r.best_for,
                )
                for r in response.recommendations
            ]
        else:
            recommendations = []

        comparison_table = ComparisonTable(
            columns=response.comparison_table.columns,
            rows=response.comparison_table.rows,
        )

        return Report(
            query="",
            title=response.title,
            direct_answer=response.direct_answer,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=recommendations,
            comparison_table=comparison_table,
            tradeoffs_and_limitations=response.tradeoffs_and_limitations,
            coverage_gaps=self._build_coverage_gaps(sub_questions),
        )

    def _fallback_report(
        self,
        sub_questions: list[SubQuestion],
    ) -> Report:
        sections: list[ReportSection] = []
        total_claims = sum(len(sq.claims) for sq in sub_questions)

        for sq in sub_questions:
            if sq.claims:
                grouped: dict[str, list[str]] = {}
                for c in sq.claims:
                    grouped.setdefault(c.source_domain, []).append(c.text)

                body_parts: list[str] = []
                for domain, texts in grouped.items():
                    body_parts.append(f"**From {domain}:**")
                    for t in texts:
                        body_parts.append(f"- {t}")
                body = "\n\n".join(body_parts)
            else:
                body = "No information was found for this question."

            sections.append(
                ReportSection(
                    heading=sq.text,
                    body=body,
                    citations=[c.source_url for c in sq.claims],
                )
            )

        summary_parts: list[str] = []
        for sq in sub_questions:
            if sq.claims:
                first_claim = sq.claims[0].text
                if len(first_claim) > 200:
                    first_claim = first_claim[:200] + "..."
                summary_parts.append(f"- **{sq.text}**: {first_claim}")

        if summary_parts:
            summary = (
                f"This report covers **{len(sub_questions)}** research sub-questions "
                f"using **{total_claims}** extracted factual claims from "
                f"**{len({c.source_url for sq in sub_questions for c in sq.claims})}** sources.\n\n"
                f"Key findings:\n" + "\n".join(summary_parts)
            )
        else:
            summary = (
                f"This report covers **{len(sub_questions)}** research sub-questions "
                f"using **{total_claims}** extracted factual claims from "
                f"**{len({c.source_url for sq in sub_questions for c in sq.claims})}** sources."
            )
        return Report(
            query="",
            executive_summary=summary,
            sections=sections,
            coverage_gaps=self._build_coverage_gaps(sub_questions),
        )
