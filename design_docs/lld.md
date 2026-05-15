# Low Level Design (LLD) — Object-Oriented Design

**Document Version:** 1.0  
**Status:** Draft  
**Companion Documents:** prd.md, hld.md  
**Language:** Python 3.12  
**Conventions:** PEP 8, type hints throughout, Pydantic for data models  
**Authors:** RoboSathi AI Engineering Team  
**Last Updated:** 15th May 2026 

---

## Document Conventions

- **Class diagrams** use text-based UML notation
- `+` = public, `-` = private, `#` = protected
- `→` = returns, `«interface»` = abstract base / protocol
- All data models inherit from `pydantic.BaseModel` unless noted
- All async methods prefixed with `async`

---

# Table of Contents

1. [Project 1 — Research Assistant Agent](#project-1--research-assistant-agent)
---

# Project 1 — Research Assistant Agent

## 1.1 Class Diagram

```
┌──────────────────────────────────────────────────────┐
│  «dataclass» SubQuestion                             │
│  + id: str                                           │
│  + text: str                                         │
│  + status: SubQuestionStatus                         │
│  + claims: list[ClaimChunk]                          │
│  + search_attempts: int                              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  «dataclass» ClaimChunk                              │
│  + claim_id: str                                     │
│  + text: str                                         │
│  + source_url: str                                   │
│  + source_domain: str                                │
│  + domain_authority: float  # 0.0–1.0               │
│  + extracted_at: datetime                            │
│  + sub_question_id: str                              │
│  + embedding: list[float]                            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  «dataclass» ContradictionRecord                     │
│  + topic: str                                        │
│  + value_a: str                                      │
│  + source_a: str                                     │
│  + value_b: str                                      │
│  + source_b: str                                     │
│  + resolution: ContradictionResolution               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  QueryAnalyzer                                       │
│  - llm: LLMClient                                    │
│  + analyze(query: str) → QueryAnalysis               │
│  + extract_ambiguities(query: str) → list[str]       │
│  + generate_clarification(ambigs) → str              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  SubQueryPlanner                                     │
│  - llm: LLMClient                                    │
│  + decompose(query: str) → list[SubQuestion]         │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ResultScorer                                        │
│  + score(results: list[SearchResult]) → list[scored] │
│  - _domain_authority(url: str) → float               │
│  - _snippet_relevance(snip, query: str) → float      │
│  - _freshness_score(date: datetime) → float          │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  SelectiveExtractor                                  │
│  - section_parser: HTMLSectionParser                 │
│  + extract(url: str, sub_q: SubQuestion) → list[str] │
│  - _parse_sections(html: str) → list[Section]        │
│  - _score_relevance(sect, query: str) → float        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ContradictionDetector                               │
│  + detect(claims: list[ClaimChunk])                  │
│    → list[ContradictionRecord]                       │
│  - _normalize_claim(text: str) → str                 │
│  - _compute_semantic_similarity(a, b) → float        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  LoopGuard                                           │
│  - executed: set[str]  (Redis-backed)                │
│  + check_and_register(query, sources) → bool         │
│  - _fingerprint(query: str, sources: list) → str     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  SynthesisEngine                                     │
│  - llm: LLMClient                                    │
│  - claim_store: ClaimStore                           │
│  + synthesize(sub_qs, contradictions) → Report       │
│  # NEVER allows model-generated citations            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ResearchOrchestrator  [entry point]                 │
│  - query_analyzer: QueryAnalyzer                     │
│  - planner: SubQueryPlanner                          │
│  - scorer: ResultScorer                              │
│  - extractor: SelectiveExtractor                     │
│  - contradiction_detector: ContradictionDetector     │
│  - loop_guard: LoopGuard                             │
│  - synthesizer: SynthesisEngine                      │
│  + async research(query: str) → Report               │
│  - async _process_sub_question(sq) → list[ClaimChunk]│
└──────────────────────────────────────────────────────┘
```

## 1.2 Key Algorithms

### Parallel Sub-Query Execution
```python
async def research(self, query: str) -> Report:
    analysis = self.query_analyzer.analyze(query)
    if analysis.has_ambiguities:
        clarification = await self._clarify(analysis.ambiguities)
        query = clarification.resolved_query

    sub_questions = self.planner.decompose(query)

    # Execute all sub-questions in parallel
    tasks = [self._process_sub_question(sq) for sq in sub_questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_claims = [c for r in results if not isinstance(r, Exception) for c in r]
    contradictions = self.contradiction_detector.detect(all_claims)
    return self.synthesizer.synthesize(sub_questions, contradictions)

async def _process_sub_question(self, sq: SubQuestion) -> list[ClaimChunk]:
    search_results = await self.search_api.search(sq.text)
    scored = self.scorer.score(search_results)
    top_urls = [r.url for r in scored[:8]]

    claims = []
    for url in top_urls:
        if self.loop_guard.check_and_register(sq.text, url):
            extracted = await self.extractor.extract(url, sq)
            claims.extend(extracted)
    return claims
```

### Fingerprint-Based Loop Detection
```python
def _fingerprint(self, query: str, source: str) -> str:
    content = f"{query.strip().lower()}|{source.strip().lower()}"
    return hashlib.sha256(content.encode()).hexdigest()

def check_and_register(self, query: str, source: str) -> bool:
    """Returns True if new (should proceed), False if duplicate."""
    fp = self._fingerprint(query, source)
    if fp in self.executed:
        return False   # duplicate — block
    self.executed.add(fp)
    self.redis.sadd(f"session:{self.session_id}:fps", fp)
    return True
```

## 1.3 Enum Definitions
```python
class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    ANSWERED = "answered"
    PARTIAL = "partial"
    UNANSWERABLE = "unanswerable"

class ContradictionResolution(str, Enum):
    PRESENT_BOTH = "present_both"
    DEFER_PRIMARY = "defer_to_primary"
    TEMPORAL_TIMELINE = "temporal_timeline"
```

---