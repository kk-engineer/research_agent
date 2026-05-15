# Product Requirements Document (PRD)

**Document Version:** 1.0  
**Status:** Draft  
**Authors:** RoboSathi AI Engineering Team  
**Last Updated:** 15th May 2026  

---

## Document Conventions

| Symbol | Meaning |
|--------|---------|
| `[MUST]` | Non-negotiable requirement |
| `[SHOULD]` | Strong preference, deviation requires justification |
| `[MAY]` | Optional, nice-to-have |
| `FR-XX` | Functional Requirement identifier |
| `NFR-XX` | Non-Functional Requirement identifier |

---

# Table of Contents
1. [Project 1 — Research Assistant Agent](#project-1--research-assistant-agent)
---

# Project 1 — Research Assistant Agent

## 1.1 Overview

A single-agent research system that accepts natural language questions, autonomously retrieves and synthesizes information from the web, and returns a structured, citation-grounded report. This project establishes the baseline architecture extended by all subsequent projects.

## 1.2 Problem Statement

Knowledge workers spend 3–5 hours per research task navigating multiple sources, evaluating credibility, resolving conflicting information, and writing summaries. Existing LLM tools either hallucinate citations or fail to surface the most authoritative sources. This agent must match or exceed the quality of a trained research analyst while operating autonomously in under 5 minutes.

## 1.3 Goals

- Automate multi-source research and synthesis for structured topic questions
- Produce reports that are citation-grounded (every claim traceable to a verified source)
- Handle ambiguous queries by clarifying before searching
- Detect and surface source contradictions rather than silently resolving them

## 1.4 Non-Goals

- Real-time data feeds or financial market data (Project 2)
- Multi-agent parallelism (Project 6)
- Persistent memory across sessions (Project 8)
- Fine-grained document analysis beyond web sources

## 1.5 User Stories

| ID | As a... | I want to... | So that... |
|----|---------|-------------|-----------|
| US-1.1 | Knowledge worker | Ask a complex research question in plain English | I don't have to perform manual web research |
| US-1.2 | Analyst | Receive a structured report with citations | I can verify every claim independently |
| US-1.3 | User | Be asked clarifying questions on ambiguous queries | I get an answer to my actual question, not a guess |
| US-1.4 | Researcher | See when two sources contradict each other | I can make an informed judgment on the conflict |
| US-1.5 | User | Understand what the agent couldn't find | I know where the gaps in the research are |

## 1.6 Functional Requirements

### Query Processing
- `FR-1.1` [MUST] The system SHALL accept natural language queries up to 2,000 characters
- `FR-1.2` [MUST] The system SHALL identify ambiguous dimensions (scope, geography, time period) and ask exactly one clarifying question before searching
- `FR-1.3` [MUST] The system SHALL decompose a resolved query into 3–5 independent sub-questions and search each in parallel

### Source Retrieval
- `FR-1.4` [MUST] The system SHALL retrieve at least 10 search results per sub-question and score them by domain authority, snippet relevance, and freshness before reading
- `FR-1.5` [MUST] The system SHALL extract only semantically relevant sections from source pages — never pass full page HTML to the model
- `FR-1.6` [MUST] The system SHALL detect duplicate sources across sub-questions and de-duplicate before synthesis
- `FR-1.7` [SHOULD] The system SHOULD prefer primary sources (government, academic, official corporate) over aggregator sites

### Conflict Detection
- `FR-1.8` [MUST] The system SHALL detect when two sources provide different values for the same claim and create a `ContradictionRecord`
- `FR-1.9` [MUST] All contradictions MUST be surfaced in the final report with both values and both source URLs — never silently resolved

### Loop Prevention
- `FR-1.10` [MUST] The system SHALL hash each (sub-query, executed-source-set) pair and refuse to re-execute identical searches
- `FR-1.11` [MUST] The system SHALL enforce a maximum of 25 tool calls per research session

### Output
- `FR-1.12` [MUST] Every claim in the final report MUST be traceable to a specific URL and extracted text chunk stored in the claim store — no model-generated citations
- `FR-1.13` [MUST] The report MUST include a coverage section listing: sub-questions answered, partially answered, and unanswerable
- `FR-1.14` [SHOULD] The report SHOULD be structured with sections: Executive Summary, Findings by Sub-Question, Contradictions, Coverage Gaps, References

## 1.7 Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1.1 | Latency | End-to-end report generation < 5 minutes for a 5-sub-question query |
| NFR-1.2 | Accuracy | Citation accuracy ≥ 95% (verified URL exists + claim present in source) |
| NFR-1.3 | Coverage | ≥ 80% of sub-questions answered or explicitly flagged unanswerable |
| NFR-1.4 | Cost | Maximum $0.50 per research session at current API pricing |
| NFR-1.5 | Availability | System uptime ≥ 99.5% |
| NFR-1.6 | Hallucination | Zero tolerance for fabricated citations |

## 1.8 Success Metrics (KPIs)

- **Task Completion Rate:** % of queries resulting in a report with ≥ 80% sub-questions answered
- **Citation Accuracy:** % of cited URLs that exist and contain the attributed claim
- **Contradiction Detection Rate:** % of known conflicts surfaced (measured against gold test set)
- **Avg Session Cost:** Mean token cost per completed research session
- **Loop Detection Rate:** % of duplicate search attempts blocked before execution

## 1.9 Constraints & Assumptions

- Assumes access to a web search API (Tavily or Duckduckgo)
- Assumes target sources are publicly accessible (no paywall scraping)
- Assumes user can accept 2–5 minute latency for complex queries
- Model context window: 4k-16k tokens 

---