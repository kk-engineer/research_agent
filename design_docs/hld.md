# High Level Design (HLD)
**Document Version:** 1.0  
**Status:** Draft  
**Companion Documents:** prd.md, lld.md  
**Authors:** RoboSathi AI Engineering Team  
**Last Updated:** 15th May 2026 

---

## Document Conventions

- **C4 Levels Used:** Context → Container → Component (System sequence diagrams where helpful)
- Architecture diagrams use ASCII/text notation for portability
- Technology choices represent production-grade defaults; project teams may substitute equivalents
- All costs are estimates based on 2025 API pricing

---

# Table of Contents

1. [Project 1 — Research Assistant Agent](#project-1--research-assistant-agent)
---

# Project 1 — Research Assistant Agent

## 1.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BOUNDARY                        │
│                                                             │
│   [User]  ──── natural language query ────►  [Research     │
│           ◄─── structured report ──────────   Assistant    │
│                                                System]      │
└─────────────────────────────────────────────────────────────┘
            External Dependencies:
            ├── Web Search API (SerpAPI / Tavily)
            ├── LLM API (Anthropic / OpenAI)
            └── Web Fetch (Playwright / requests)
```

## 1.2 Container Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Research Assistant System                           │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────────┐ │
│  │  CLI / API  │───►│   Agent     │───►│          Tool Registry           │ │
│  │  Interface  │    │ Orchestrator│    │  ┌──────────┐  ┌──────────────┐  │ │
│  └─────────────┘    └──────┬──────┘    │  │Web Search│  │  Page Fetch  │  │ │
│                            │           │  └──────────┘  └──────────────┘  │ │
│                     ┌──────▼──────┐    │  ┌──────────┐  ┌──────────────┐  │ │
│                     │  Working    │    │  │  Chunk   │  │  Citation    │  │ │
│                     │  Memory     │    │  │Extractor │  │  Validator   │  │ │
│                     │  Store      │    │  └──────────┘  └──────────────┘  │ │
│                     │  (Redis)    │    └──────────────────────────────────┘ │
│                     └─────────────┘                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Claim & Evidence Store (Redis)    │   │
│  │  search_records | claim_chunks | contradiction_records | coverage_map│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 1.3 Data Flow (Request → Report)

```
1. User submits query
        │
2. Query Analyzer extracts intent, scope, ambiguities
        │
3. If ambiguities detected → Clarification Engine generates 1 question → User responds
        │
4. Parallel Sub-Query Planner decomposes into 3–5 sub-questions
        │
        ├──[async]── Sub-question 1 → Search → Score → Extract → Store Claims
        ├──[async]── Sub-question 2 → Search → Score → Extract → Store Claims
        └──[async]── Sub-question N → Search → Score → Extract → Store Claims
        │
5. Contradiction Detector compares claims across sub-questions for same metrics
        │
6. Coverage Checker reviews status of all sub-questions
        │
7. Synthesis Engine generates report from claim store ONLY (no model training data citations)
        │
8. Report Formatter structures output with sections + references
        │
9. Final report returned to user
```

## 1.4 Technology Stack

| Layer | Technology                    | Rationale |
|-------|-------------------------------|-----------|
| LLM | Claude Sonnet 4 / GPT-4o      | Best instruction following for structured output |
| Web Search | Tavily API                    | Returns clean snippets, not raw HTML |
| Page Fetch | BeautifulSoup + requests      | Lightweight HTML parsing |
| Claim Store | Redis                         | Semantic search over claims |
| Working Memory | Redis (TTL: session duration) | Fast key-value for loop detection |
| Orchestration | Python async (asyncio)        | Parallel sub-query execution |
| Output | Pydantic models → Markdown    | Structured output validation |

## 1.5 Scalability & Capacity

- Single-user CLI: no horizontal scaling needed
- Async parallel search: 3–5x speedup vs. sequential
- Redis TTL on search fingerprints: auto-expires after session
- Redis write volume: ~50 claims per session → negligible

## 1.6 Key Design Decisions

| Decision | Chosen Approach | Rejected Alternative | Reason |
|----------|----------------|---------------------|--------|
| Source reading | Section extraction only | Full page in context | Full page overflows context, dilutes relevance |
| Citation generation | Store-only (claim_store) | Model generates citations | Eliminates hallucinated URLs |
| Contradiction handling | Present both with metadata | Auto-resolve to higher authority | Preserves user agency and is legally defensible |
| Parallel search | asyncio fan-out | Sequential sub-questions | 3–5x latency improvement |

---