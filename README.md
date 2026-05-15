# Research Assistant Agent

An autonomous AI research agent that accepts natural language questions, retrieves and
synthesises information from the web, and returns structured citation-grounded reports.

Built following the design specifications in `design_docs/prd.md`, `design_docs/hld.md`,
and `design_docs/lld.md`.

---

## Features

- **Natural language queries** — ask research questions in plain English
- **Ambiguity resolution** — detects unclear queries and asks a clarifying question
- **Parallel sub-query execution** — decomposes complex queries into 3–5 sub-questions
  and researches them concurrently
- **Source scoring** — ranks search results by domain authority, snippet relevance, and
  freshness
- **Selective extraction** — uses an LLM to extract only semantically relevant sections
  from source pages
- **Contradiction detection** — identifies conflicting claims across sources and surfaces
  both sides
- **Loop prevention** — SHA-256 fingerprinting prevents repeat fetching of the same
  (query, URL) pair; 25-tool-call hard cap
- **Citation-grounded reports** — every claim is traceable to a verified source URL;
  never uses model-generated citations
- **Coverage gaps** — explicitly lists what could and could not be answered
- **Local model support** — works with [llama.cpp](https://github.com/ggerganov/llama.cpp)
  and any OpenAI-compatible local server
- **Live progress display** — real-time phase tracking with elapsed time via
  [Rich](https://github.com/Textualize/rich)
- **Structured logging** — configurable log levels, file output, and rich tracebacks

---

## Architecture

```
src/
├── main.py                  # CLI entry point (single query & interactive modes)
├── config.py                # Environment-based configuration
├── models.py                # Pydantic data models
├── prompts.py               # Centralised LLM system prompts
├── logger.py                # Structured logging setup (Rich + file)
├── progress.py              # Live progress tracker with timing
├── utils.py                 # Shared utilities (JSON extraction, etc.)
├── report_formatter.py      # Report → Markdown formatting
├── llm_client.py            # LLM abstraction (OpenAI / llama.cpp / mock)
├── web_search.py            # Search providers (Tavily / DuckDuckGo)
├── page_fetcher.py          # HTTP page fetching
├── text_processor.py        # HTML → text extraction, chunking, domain parsing
├── query_analyzer.py        # Query intent & ambiguity analysis
├── sub_query_planner.py     # Query decomposition into sub-questions
├── result_scorer.py         # Domain authority & relevance scoring
├── selective_extractor.py   # LLM-guided relevant claim extraction
├── contradiction_detector.py# Cross-source contradiction detection
├── loop_guard.py            # Duplicate search fingerprinting
├── claim_store.py           # Claim storage with deduplication
├── synthesis_engine.py      # Report generation from claim store
└── orchestrator.py          # Main research orchestration (async)
```

---

## Requirements

- **Python 3.12+**
- **API keys** (depending on providers):
  - [OpenAI](https://platform.openai.com/api-keys) API key (or compatible provider)
  - [Tavily](https://tavily.com) API key (optional — DuckDuckGo fallback available)
- **Or** a local [llama.cpp](https://github.com/ggerganov/llama.cpp) server

---

## Setup

### 1. Clone and navigate

```bash
cd research_agent
```

### 2. Create a virtual environment

```bash
uv venv
```

```bash
source .venv/bin/activate
```

> Windows: `.venv\Scripts\activate`

### 3. Install with uv (recommended)

```bash
uv sync
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
LLM_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-api-key
```

To use DuckDuckGo instead of Tavily (no API key needed):

```env
SEARCH_PROVIDER=duckduckgo
```

### 5. Verify configuration

```bash
python src/main.py --validate-config
```

---

## Usage

### Interactive Shell (Enter research topic/user query)

```bash
 python src/main.py -i
```

The report is automatically saved to `research_reports/{slug}_research_report.md`
(e.g. `research_reports/what_are_the_latest_advances_in_mrna_vaccine_research_report.md`).
A clickable `file://` link is displayed in the terminal.

### Save to a specific path

```bash
python src/main.py "Explain quantum error correction" --output my_report.md
```

### Copy to clipboard

```bash
python src/main.py "What is the heat capacity of water?" --copy
```

### Skip auto-save (interactive prompt after report)

```bash
python src/main.py "Compare RISC-V and ARM" --no-save
```

After the report displays, you'll be prompted to save, copy, or skip:

```
[save/copy/both/skip] s
```

### Interactive mode

```bash
python src/main.py --interactive
```

Type queries at the `research>` prompt. After each report, you're prompted to
save, copy to clipboard, or skip.

Commands:

| Command   | Action           |
|-----------|------------------|
| `/quit`   | Exit             |
| `/help`   | Show help        |
| `/clear`  | Clear the screen |

### Verbose / debug logging

```bash
python src/main.py "query" --verbose
```

### Skip clarification

```bash
python src/main.py "Compare RISC-V and ARM" --skip-clarification
```

---

## Live Progress Display

When running a query, a live panel shows each research phase with elapsed time:

```
┌────────────────────────────────────────────────────────┐
│                   Research Progress                     │
├────────────────────────────────────────────────────────┤
│ DONE  Query analysis                     [2.3s]        │
│ DONE  Sub-question planning              [1.1s]        │
│ DONE  Sub-Q: What are the key...         [8.7s]  3 clm│
│ DONE  Sub-Q: Who are the leading...      [7.2s]  5 clm│
│ DONE  Deduplication                      [0.0s] rem 0 │
│ DONE  Contradiction detection            [3.1s]  1 fnd│
│ DONE  Report synthesis                   [5.4s]  8 clm│
│ ... Total: 28.0s                                        │
└────────────────────────────────────────────────────────┘
```

---

## Using Local Models (llama.cpp)

### 1. Download a model

```bash
git clone https://github.com/ggerganov/llama.cpp
```

```bash
cd llama.cpp && make
```

Download a model (e.g. Qwen 2.5 7B Instruct):

```bash
wget -O models/qwen2.5-7b-instruct.Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct.Q4_K_M.gguf
```

### 2. Start the server

```bash
./llama-server -m models/qwen2.5-7b-instruct.Q4_K_M.gguf \
  --port 8080 \
  --n-gpu-layers -1    # offload all layers to GPU if available
```

### 3. Configure `.env`

```env
LLM_PROVIDER=llamacpp
LLAMA_BASE_URL=http://localhost:8080/v1
LLAMA_MODEL=local-model
```

### 4. Run the agent

```bash
python src/main.py "What is the heat capacity of water?"
```

The agent communicates with llama.cpp through its OpenAI-compatible API endpoint,
so no separate API key is required.

---

## CLI Reference

| Argument / Flag          | Description                                    |
|--------------------------|------------------------------------------------|
| `query`                  | Research question (positional, optional)       |
| `--interactive`, `-i`    | Interactive chat mode                          |
| `--output`, `-o`         | Path to save the report markdown               |
| `--copy`, `-c`           | Copy the report to clipboard automatically     |
| `--no-save`              | Skip auto-saving (prompts after report)        |
| `--skip-clarification`   | Don't ask clarifying questions                 |
| `--validate-config`      | Check configuration and exit                   |
| `--verbose`, `-v`        | Enable debug logging                           |

---

## Configuration Reference

| Variable               | Default                  | Description                               |
|------------------------|--------------------------|-------------------------------------------|
| `LLM_PROVIDER`         | `openai`                 | `openai`, `llamacpp`, or `mock`           |
| `LLM_MODEL`            | `gpt-4o`                 | Model name                                |
| `LLM_API_KEY`          | —                        | API key for OpenAI                        |
| `LLM_BASE_URL`         | —                        | Custom base URL for OpenAI-compatible API |
| `LLAMA_BASE_URL`       | `http://localhost:8080/v1`| llama.cpp server URL                      |
| `LLAMA_MODEL`          | `local-model`            | Model label for llama.cpp                 |
| `EMBEDDING_PROVIDER`   | —                        | Embedding provider (defaults to LLM)      |
| `EMBEDDING_MODEL`      | —                        | Embedding model name                      |
| `EMBEDDING_BASE_URL`   | —                        | Embedding server URL                      |
| `SEARCH_PROVIDER`      | `tavily`                 | `tavily` or `duckduckgo`                  |
| `TAVILY_API_KEY`       | —                        | API key for Tavily search                 |
| `LLM_MAX_TOKENS`        | `2048`                   | Default max tokens per LLM call (overridable per-component) |
| `STATE_TIMEOUT`         | `45`                     | Per-phase timeout (seconds); agent skips phase if exceeded |
| `LLM_REQUEST_TIMEOUT`   | `30`                     | Max wait for a single LLM API call        |
| `PAGE_FETCH_TIMEOUT`    | `10`                     | Max wait for a single page fetch          |
| `MAX_SEARCH_RESULTS`    | `5`                      | Results per sub-query search (= 1 Tavily call per sub-question) |
| `TOP_URLS_TO_FETCH`     | `5`                      | Top-URLs to fetch per sub-query           |
| `MAX_CHUNKS_PER_PAGE`   | `3`                      | Text chunks analysed per page             |
| `MAX_CONTRADICTION_PAIRS` | `5`                    | Max contradiction pairs checked via LLM   |
| `MAX_TOOL_CALLS`        | `25`                     | Hard cap on tool calls per session        |
| `MAX_SUB_QUESTIONS`     | `3`                      | Maximum sub-questions                     |
| `MIN_SUB_QUESTIONS`     | `2`                      | Minimum sub-questions                     |
| `LOG_LEVEL`            | `INFO`                   | `DEBUG`, `INFO`, `WARNING`, `ERROR`       |
| `LOG_FILE`             | `agent.log`              | File path for persistent debug logs       |

---

## Report Structure

Each report includes:

1. **Executive Summary** — high-level overview of findings
2. **Findings by Sub-Question** — detailed findings with citations
3. **Contradictions** — conflicting claims with both values and source URLs
4. **Coverage Gaps** — sub-questions that could not be fully answered
5. **References** — numbered list of all source URLs cited

---

## Development

### Formatting & linting

```bash
pip install ruff
```

```bash
ruff check src/
```

```bash
ruff format src/
```

### Adding dependencies

```bash
uv add some-package
# Or add manually to pyproject.toml [project.dependencies]
```

---

## Design Documentation

Refer to the `design_docs/` directory:

- **PRD** — product requirements, user stories, functional & non-functional requirements
- **HLD** — system context, container architecture, data flow, technology stack
- **LLD** — class diagram, key algorithms, data models, enum definitions

---

## License

Proprietary — RoboSathi AI Engineering Team
