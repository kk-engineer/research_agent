from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt

from src.claim_store import ClaimStore, InMemoryClaimStore
from src.config import settings
from src.contradiction_detector import ContradictionDetector
from src.debug_utils import print_debug_info
from src.llm_client import get_llm_client, LLMClient
from src.logger import AgentLogger, begin_step, end_step, log_json_event, log_agent_state, log_intermediate_step
from src.loop_guard import LoopGuard
from src.metrics import metrics
from src.models import ClaimChunk, ContradictionRecord, CoverageGap, Report, SubQuestion, SubQuestionStatus
from src.progress import ResearchProgress
from src.query_analyzer import QueryAnalyzer
from src.result_scorer import ResultScorer
from src.selective_extractor import SelectiveExtractor
from src.sub_query_planner import SubQueryPlanner
from src.synthesis_engine import SynthesisEngine
from src.telemetry import timed_async
from src.text_processor import extract_domain
from src.tracing import tracer
from src.utils import with_timeout
from src.web_search import get_search_provider, WebSearchProvider
logger = logging.getLogger(__name__)

T = settings.state_timeout


class ResearchOrchestrator:
    def __init__(
        self,
        llm: LLMClient | None = None,
        search: WebSearchProvider | None = None,
        claim_store: ClaimStore | None = None,
        loop_guard: LoopGuard | None = None,
        console: Console | None = None,
    ) -> None:
        self._llm = llm or get_llm_client()
        self._search = search or get_search_provider()
        self._claim_store = claim_store or InMemoryClaimStore()
        self._loop_guard = loop_guard or LoopGuard()
        self._console = console or Console()

        self._query_analyzer = QueryAnalyzer(self._llm)
        self._planner = SubQueryPlanner(self._llm)
        self._scorer = ResultScorer()
        self._extractor = SelectiveExtractor(self._llm)
        self._contradiction_detector = ContradictionDetector(self._llm)
        self._synthesizer = SynthesisEngine(self._llm)

    async def research(
        self, query: str, skip_clarification: bool = False
    ) -> Report:
        self._loop_guard.reset()
        self._claim_store.clear()

        progress = ResearchProgress(self._console, use_live=True)
        progress.start()
        self._llm.on_token = progress.add_stream_token
        progress.log("🧠 Research session started", "thought")
        progress.log(f"📝 Query: {query}", "info")

        total_start = time.monotonic()

        log_agent_state("QUERY_RECEIVED", f"User query: {query}")
        log_json_event("research_start", {"query": query, "session_id": settings.session_id})

        try:
            # ── 1. Query Analysis ─────────────────────────────────
            log_agent_state("ANALYZING", f"Analyzing query: {query}")
            begin_step("Query analysis")
            progress.set_tool("LLM / QueryAnalyzer")
            progress.set_action("Analyzing query for intent, scope, ambiguities…")
            progress.log("🧠 Thought: Need to understand what the user is asking", "thought")
            progress.log(
                "   Before searching, extract intent, scope, geography, time period",
                "data",
            )
            progress.log(
                AgentLogger.llm_call("QueryAnalyzer.analyze", getattr(self._llm, '_model', '')),
                "llm",
            )
            progress.phase("Query analysis")

            tracer.event("Query analysis start", "analyze")
            async with timed_async("query_analysis"):
                analysis = await self._query_analyzer.analyze(query)
            tracer.complete_event(
                "Query analysis complete", "analyze",
                time.monotonic() - total_start,
            )

            if analysis is None:
                from src.models import QueryAnalysis
                analysis = QueryAnalysis(
                    original_query=query,
                    primary_intent="unknown",
                    scope="broad",
                    ambiguities=[],
                    has_ambiguities=False,
                )
                progress.log("⚠ Query analysis failed — using defaults", "warn")
            progress.add_tokens(len(query) // 4)
            progress.mark_done("Query analysis")

            had_ambiguities = analysis.has_ambiguities
            progress.log(
                f"   Scope: {analysis.scope}  |  Intent: {analysis.primary_intent}",
                "data",
            )
            if had_ambiguities:
                progress.log(f"   Ambiguities: {len(analysis.ambiguities)} dimension(s)", "data")
            log_intermediate_step("Query Analysis Result",
                f"Intent: {analysis.primary_intent}\n"
                f"Scope: {analysis.scope}\n"
                f"Ambiguities: {len(analysis.ambiguities)}\n"
                f"Constraints: {analysis.constraints}" if hasattr(analysis, 'constraints') else "")
            end_step()

            # ── 2. Clarification (reflection / human-in-the-loop) ─
            resolved_query = query
            if had_ambiguities and not skip_clarification:
                log_agent_state("CLARIFYING", f"Query has {len(analysis.ambiguities)} ambiguous dimensions")
                begin_step("Clarification")
                progress.set_tool("LLM / ClarificationEngine")
                progress.set_action("Resolving ambiguities…")
                progress.log("🧠 Thought: Query has ambiguous dimensions", "thought")
                progress.phase("Query clarification")

                clarification = await self._query_analyzer.generate_clarification(
                    query, analysis.ambiguities
                )

                for amb in analysis.ambiguities:
                    progress.log(f"   · Ambiguity: {amb}", "data")

                if clarification.needs_clarification and clarification.question:
                    progress.log(f"   Asking user: {clarification.question}", "data")
                    answer = Prompt.ask(f"[bold cyan]{clarification.question}[/bold cyan]")
                    if answer.strip():
                        resolved_query = f"{clarification.resolved_query or query}. User clarified: {answer.strip()}"
                        progress.log(f"   User resolved → {answer.strip()}", "done")
                    else:
                        resolved_query = clarification.resolved_query or query
                        progress.log("   No answer — using LLM-resolved query", "data")
                else:
                    resolved_query = clarification.resolved_query or query
                    progress.log(f"   Resolved via reflection → {resolved_query}", "done")
                    if clarification.assumptions:
                        for a in clarification.assumptions:
                            progress.log(f"     assumption: {a}", "info")

                progress.mark_done("Query clarification")
                end_step()

            # ── 3. Sub-question planning ─────────────────────────
            log_agent_state("PLANNING", "Decomposing query into sub-questions")
            begin_step("Sub-question planning")
            progress.set_tool("LLM / SubQueryPlanner")
            progress.set_action("Decomposing query into research sub-questions…")
            progress.log("🧠 Thought: Decompose into independent sub-questions for parallel research", "thought")
            progress.log(AgentLogger.llm_call("SubQueryPlanner.decompose"), "llm")
            progress.phase("Sub-question planning")

            tracer.event("Sub-question planning start", "plan")
            async with timed_async("sub_question_planning"):
                sub_questions = await self._planner.decompose(resolved_query)
            tracer.complete_event(
                f"Planned {len(sub_questions) if sub_questions else 0} sub-questions",
                "plan",
                time.monotonic() - total_start,
            )

            if not sub_questions:
                n = max(settings.min_sub_questions, 1)
                sub_questions = [
                    SubQuestion(text=f"What are the key facts about {query}?") for _ in range(n)
                ]
            progress.add_tokens(300)
            progress.mark_done("Sub-question planning")
            progress.log(
                f"   Planned {len(sub_questions)} sub-questions for parallel research", "data"
            )
            for i, sq in enumerate(sub_questions, 1):
                progress.log(f"   [{i}] {sq.text}", "data")
            sq_texts = "\n".join(f"{i}. {sq.text}" for i, sq in enumerate(sub_questions, 1))
            log_intermediate_step("Sub-Questions Planned",
                f"Total: {len(sub_questions)}\n\n{sq_texts}")
            log_json_event("sub_questions_planned", {
                "count": len(sub_questions),
                "questions": [sq.text for sq in sub_questions],
            })
            end_step()

            # ── 4. Sub-question research (parallel) ──────────────
            log_agent_state("RESEARCHING", f"Researching {len(sub_questions)} sub-questions in parallel")
            async def _run_sub_q(sq: SubQuestion) -> list[ClaimChunk]:
                sq_label = f"Sub-Q: {sq.text}"
                progress.phase(sq_label)
                tracer.push_level()
                try:
                    async with timed_async(f"sub_question_{sq.id}"):
                        claims = await with_timeout(
                            self._process_sub_question(sq, progress),
                            timeout_sec=T * 2,
                            label=f"sub_q_{sq.id}",
                        )
                    if claims is None:
                        claims = []
                    sq.claims = claims
                    sq.status = (
                        SubQuestionStatus.ANSWERED
                        if claims
                        else SubQuestionStatus.UNANSWERABLE
                    )
                    if not settings.disable_claims:
                        self._claim_store.add_claims(claims)
                    detail = f"{len(claims)} clm"
                    progress.mark_done(sq_label, detail)
                    if claims:
                        label_text = "snippet(s)" if settings.disable_claims else "claim(s)"
                        progress.log(
                            f"✔ Sub-question done — {len(claims)} {label_text}", "done"
                        )
                    else:
                        progress.log("⚠ No results for this sub-question", "warn")
                    tracer.pop_level()
                    return claims
                except Exception as e:
                    sq.status = SubQuestionStatus.UNANSWERABLE
                    progress.mark_failed(sq_label, str(e))
                    progress.log(f"⚠ Sub-question failed", "warn")
                    logger.warning("Sub-question failed: %s", e)
                    tracer.pop_level()
                    return []

            tasks = [_run_sub_q(sq) for sq in sub_questions]
            await asyncio.gather(*tasks, return_exceptions=True)

            all_claims: list[ClaimChunk] = []
            for sq in sub_questions:
                all_claims.extend(sq.claims)

            all_urls = {c.source_url for c in all_claims}
            if not settings.disable_claims:
                metrics.claims_extracted = len(all_claims)
                progress.log(f"📊 Total claims collected: {len(all_claims)}", "data")
                log_json_event("claims_collected", {
                    "total": len(all_claims),
                    "unique_sources": len(all_urls),
                })

            # ── 5. Deduplication (skipped when claims disabled) ──
            if not settings.disable_claims:
                log_agent_state("DEDUPLICATING", f"Deduplicating {len(all_claims)} claims")
                begin_step("Deduplication")
                progress.set_tool("ClaimStore")
                progress.set_action("Deduplicating extracted claims…")
                progress.log(
                    f"🧠 Thought: Deduplicate {len(all_claims)} claim(s) from multiple sources",
                    "thought",
                )
                progress.phase("Deduplication")
                tracer.event("Deduplication start", "dedup")
                before = len(self._claim_store.get_all_claims())
                removed = self._claim_store.deduplicate()
                after = len(self._claim_store.get_all_claims())
                metrics.claims_deduped = removed
                tracer.complete_event(
                    f"Dedup: {before} → {after} ({removed} removed)", "dedup",
                    time.monotonic() - total_start,
                )
                progress.mark_done("Deduplication", f"removed {removed}")
                progress.log(f"   {before} → {after}  ({removed} duplicate(s) removed)", "data")
                end_step()

            # ── 6. Contradiction detection (skipped when claims disabled) ──
            contradictions: list[ContradictionRecord] = []
            if not settings.disable_claims:
                log_agent_state("DETECTING_CONTRADICTIONS", f"Scanning {len(all_claims)} claims")
                begin_step("Contradiction detection")
                progress.set_tool("LLM / ContradictionDetector")
                progress.set_action(f"Scanning {len(all_claims)} claims for contradictions…")
                progress.log("🧠 Thought: Check if different sources disagree on same facts", "thought")
                progress.log(
                    f"   Checking {len(all_claims)} claims across sources for conflicts", "data"
                )
                progress.phase("Contradiction detection")

                tracer.event("Contradiction detection start", "contradiction")
                async with timed_async("contradiction_detection"):
                    contradictions = await self._contradiction_detector.detect(
                        all_claims
                    )
                tracer.complete_event(
                    f"Contradictions: {len(contradictions) if contradictions else 0} found",
                    "contradiction",
                    time.monotonic() - total_start,
                )

                if contradictions is None:
                    contradictions = []
                progress.add_tokens(len(all_claims) * 50)
                progress.mark_done("Contradiction detection", f"{len(contradictions)} found")
                if contradictions:
                    progress.log(AgentLogger.contradiction(len(contradictions)), "warn")
                    for c in contradictions:
                        progress.log(f"   ⚡ {c.topic[:60]}", "warn")
                else:
                    progress.log("   No contradictions found across sources", "data")
                end_step()

            # ── 7. Report synthesis ──────────────────────────────
            log_agent_state("SYNTHESIZING", f"Generating report from {len(all_claims)} claims")
            begin_step("Report synthesis")
            progress.clear_stream()
            progress.set_tool("LLM / SynthesisEngine")
            progress.set_action("Generating structured research report…")
            source_type = "snippets" if settings.disable_claims else "claims"
            progress.log("🧠 Thought: Synthesize all verified claims into a structured report", "thought")
            progress.log(AgentLogger.llm_call("SynthesisEngine.synthesize"), "llm")
            progress.log(f"   Generating report from {len(all_claims)} {source_type}", "data")
            progress.phase("Report synthesis")

            tracer.event("Report synthesis start", "synthesize")
            async with timed_async("report_synthesis"):
                report = await self._synthesizer.synthesize(
                    sub_questions, contradictions
                )
            tracer.complete_event(
                "Report synthesis complete", "synthesize",
                time.monotonic() - total_start,
            )

            if report is None or not any(s.body.strip() for s in report.sections):
                from src.models import Report as ReportModel, ReportSection as ReportSectionModel
                fallback_sections = [
                    ReportSectionModel(
                        heading=sq.text,
                        body="\n\n".join(
                            f"**{c.source_domain}:** {c.text}" for c in sq.claims
                        ) if sq.claims else "No information was found for this question.",
                        citations=[c.source_url for c in sq.claims],
                    )
                    for sq in sub_questions
                ]
                all_urls = sorted({c.source_url for sq in sub_questions for c in sq.claims})
                answered_count = sum(1 for sq in sub_questions if sq.status == SubQuestionStatus.ANSWERED)
                source_label = "snippets" if settings.disable_claims else "claims"

                summary_parts: list[str] = []
                for sq in sub_questions:
                    if sq.claims:
                        first_claim = sq.claims[0].text
                        if len(first_claim) > 200:
                            first_claim = first_claim[:200] + "..."
                        summary_parts.append(f"- **{sq.text}**: {first_claim}")

                if summary_parts:
                    executive_summary = (
                        f"This report covers **{len(sub_questions)}** research sub-questions "
                        f"using **{len(all_claims)}** {source_label} from **{len(all_urls)}** sources."
                        + (f" **{answered_count}** sub-question(s) were fully answered." if answered_count else "")
                        + "\n\nKey findings:\n" + "\n".join(summary_parts)
                    )
                else:
                    executive_summary = (
                        f"Research covered **{len(sub_questions)}** sub-question(s) "
                        f"using **{len(all_claims)}** {source_label} from **{len(all_urls)}** sources."
                        + (f" **{answered_count}** sub-question(s) were fully answered." if answered_count else "")
                    )

                report = ReportModel(
                    query=resolved_query,
                    executive_summary=executive_summary,
                    sections=fallback_sections,
                    contradictions=contradictions,
                    coverage_gaps=[gs for sq in sub_questions for gs in (
                        [] if sq.claims else [
                            CoverageGap(
                                sub_question_id=sq.id,
                                sub_question_text=sq.text,
                                status=SubQuestionStatus.UNANSWERABLE,
                                note="No supporting information found from any source.",
                            )
                        ]
                    )],
                    references=all_urls,
                )
                progress.log("⚠ Report synthesis timed out or empty — using fallback", "warn")

            report.query = resolved_query
            report.generated_at = datetime.utcnow()
            report.web_search_calls = metrics.search_count
            progress.add_tokens(2000)
            source_label = "snippet(s)" if settings.disable_claims else "claim(s)"
            progress.mark_done("Report synthesis", f"{len(all_claims)} {source_label}")

            total_dur = time.monotonic() - total_start
            progress.log(
                AgentLogger.report(len(all_claims), len(report.references), total_dur),
                "done",
            )

            log_agent_state("COMPLETE", f"Report generated in {total_dur:.1f}s")
            log_json_event("research_complete", {
                "total_claims": len(all_claims),
                "total_sources": len(report.references),
                "total_duration": round(total_dur, 2),
                "total_tokens": metrics.total_tokens,
                "total_cost": metrics.total_cost,
                "llm_calls": metrics.llm_call_count,
                "searches": metrics.search_count,
            })
            end_step()

            return report

        finally:
            progress.clear_stream()
            self._llm.on_token = None
            progress.stop()

    async def close(self) -> None:
        await self._search.close()

    async def _process_sub_question(
        self, sub_question: SubQuestion, progress: ResearchProgress
    ) -> list[ClaimChunk]:
        query_preview = sub_question.text

        # ── Search ─────────────────────────────────────────────
        log_agent_state("SEARCHING", f"Sub-Q: {query_preview[:80]}")
        begin_step(f"Search: {query_preview[:60]}")
        progress.set_tool(f"Search / {settings.search_provider}")
        progress.set_action(f"Searching the web for '{query_preview[:80]}'…")
        progress.log(
            AgentLogger.action(
                settings.search_provider, f"query='{query_preview}'"
            ),
            "tool",
        )

        async with timed_async(f"search_{sub_question.id}"):
            search_results = await with_timeout(
                self._search.search(
                    sub_question.text,
                    max_results=settings.max_search_results_per_query,
                ),
                timeout_sec=T,
                label=f"search_{sub_question.id}",
            )
        if not search_results:
            end_step("no results")
            return []
        progress.add_tokens(len(query_preview) // 4)
        progress.add_sources(len(search_results))
        progress.log(f"   Top results:", "data")
        for i, r in enumerate(search_results[:3], 1):
            progress.log(f"   [{i}] {r.title[:55]} — {r.url[:50]}", "data")
        end_step()

        if settings.disable_claims:
            now = datetime.utcnow()
            claims: list[ClaimChunk] = []
            for r in search_results:
                snippet_text = r.snippet or r.title or ""
                if len(snippet_text.strip()) < settings.fast_mode_min_snippet_length:
                    continue
                claims.append(
                    ClaimChunk(
                        text=snippet_text.strip(),
                        source_url=r.url,
                        source_domain=extract_domain(r.url),
                        domain_authority=settings.fast_mode_domain_authority,
                        extracted_at=now,
                        sub_question_id=sub_question.id,
                    )
                )
            if claims:
                progress.log(
                    f"   Used {len(claims)} snippet(s) from {len(search_results)} result(s)",
                    "data",
                )
            return claims

        # ── Score → pick top URLs ──────────────────────────────
        scored = self._scorer.score(search_results, sub_question.text)
        top_urls = [r[0].url for r in scored[: settings.top_urls_to_fetch]]
        progress.log(
            f"   Selected {len(top_urls)} top URLs by authority + relevance", "data"
        )

        # ── Fetch + Extract per URL (parallel) ─────────────────
        progress.set_tool(f"Extracting from {len(top_urls)} sources...")
        progress.set_action(f"Extracting claims from {len(top_urls)} sources...")

        async def _extract_one(url: str, idx: int) -> list[ClaimChunk]:
            if not self._loop_guard.check_and_register(sub_question.text, url):
                progress.log(f"   ⏭ Skipped (already processed)", "data")
                return []

            short_url = url.replace("https://", "").replace("http://", "")[:50]

            async with timed_async(f"extract_{url[:40]}"):
                extracted = await self._extractor.extract(url, sub_question)

            if extracted:
                metrics.page_fetches += 1
                progress.add_claims(len(extracted))
                progress.log(AgentLogger.extract(short_url, len(extracted)), "done")
            else:
                progress.log(AgentLogger.extract(short_url), "data")
                metrics.page_fetch_errors += 1
            sub_question.search_attempts += 1
            progress.add_tokens(200)
            return extracted or []

        tasks = [_extract_one(url, i) for i, url in enumerate(top_urls, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        claims: list[ClaimChunk] = []
        for r in results:
            if isinstance(r, list):
                claims.extend(r)

        progress.log(
            f"   Sub-question total: {len(claims)} claim(s) from {len(top_urls)} URLs",
            "data",
        )
        return claims
