from __future__ import annotations

from src.models import Report


def format_report(report: Report) -> str:
    lines: list[str] = []

    title = report.title or "Research Report"
    lines.append(f"# {title}\n")
    lines.append(f"**Query:** {report.query}\n")
    lines.append(f"**Generated:** {report.generated_at.isoformat()}\n")
    lines.append(f"**Sources consulted:** {len(report.references)}")
    lines.append(f"**Web search calls:** {report.web_search_calls}")
    lines.append("")

    if report.direct_answer:
        lines.append("## Direct Answer\n")
        lines.append(report.direct_answer)
        lines.append("")

    if report.executive_summary:
        lines.append("## Executive Summary\n")
        lines.append(report.executive_summary)
        lines.append("")

    if report.recommendations:
        lines.append("## Recommendations\n")
        for rec in report.recommendations:
            lines.append(f"### {rec.name}\n")
            if rec.why_recommended:
                lines.append(f"- **Why:** {rec.why_recommended}")
            if rec.strengths:
                lines.append(f"- **Strengths:** {', '.join(rec.strengths)}")
            if rec.weaknesses:
                lines.append(f"- **Weaknesses:** {', '.join(rec.weaknesses)}")
            if rec.best_for:
                lines.append(f"- **Best for:** {rec.best_for}")
            lines.append("")

    if report.sections:
        for section in report.sections:
            lines.append(f"## {section.heading}\n")
            lines.append(section.body)
            if section.citations:
                lines.append("\n*Sources:* " + ", ".join(sorted(set(section.citations))))
            lines.append("")

    if report.comparison_table and report.comparison_table.columns:
        lines.append("## Comparison Table\n")
        header = " | ".join(report.comparison_table.columns)
        lines.append(f"| {header} |")
        lines.append("| " + " | ".join("---" for _ in report.comparison_table.columns) + " |")
        for row in report.comparison_table.rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if report.tradeoffs_and_limitations:
        lines.append("## Tradeoffs & Limitations\n")
        for item in report.tradeoffs_and_limitations:
            lines.append(f"- {item}")
        lines.append("")

    if report.contradictions:
        lines.append("## Contradictions\n")
        lines.append(
            "> The following contradictions were detected across sources. "
            "Both sides are presented without resolution.\n"
        )
        for i, c in enumerate(report.contradictions, 1):
            lines.append(f"### Contradiction {i}: {c.topic}\n")
            if c.impact:
                lines.append(f"- **Impact:** {c.impact}")
            lines.append(f"- **Source A ({c.source_a}):** {c.source_a_claim}")
            lines.append(f"- **Source B ({c.source_b}):** {c.source_b_claim}")
            lines.append(f"- **Handling:** {c.recommended_handling}\n")

    if report.coverage_gaps:
        lines.append("## Coverage Gaps\n")
        lines.append(
            "The following aspects of your query could not be fully addressed:\n"
        )
        for gap in report.coverage_gaps:
            lines.append(f"- **{gap.sub_question_text}** ({gap.status.value})")
            lines.append(f"  - {gap.note}")

    if report.references:
        lines.append("## References\n")
        for i, url in enumerate(report.references, 1):
            lines.append(f"{i}. {url}")

    return "\n".join(lines)
