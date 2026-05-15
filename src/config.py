from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _load_toml() -> dict:
    config_path = Path("config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def _toml_val(cfg: dict, *keys: str, default=None):
    d: object = cfg
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return default
    if d == {}:
        return default
    return d


_toml = _load_toml()


def _ev(key: str, default: str = "") -> str:
    return os.getenv(key, "")


def _ev_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _ev_bool(key: str) -> bool:
    return os.getenv(key, "").lower() in ("1", "true", "yes")


_VERBOSITY_MAP = {"minimal": 0, "normal": 1, "debug": 2, "trace": 3}


@dataclass
class Settings:
    # ── LLM ────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: _toml_val(_toml, "llm", "provider", default="llamacpp")
    )
    llm_model: str = field(
        default_factory=lambda: _toml_val(_toml, "llm", "model", default="local-model")
    )
    llm_api_key: str = field(default_factory=lambda: _ev("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _ev("LLM_BASE_URL", ""))
    llm_max_tokens: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "llm", "max_tokens", default="8192")
        )
    )
    llm_request_timeout: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "llm", "request_timeout", default="30")
        )
    )

    # ── llama.cpp ──────────────────────────────────────────────
    llama_base_url: str = field(
        default_factory=lambda: _toml_val(
            _toml, "llama", "base_url", default="http://localhost:8080/v1"
        )
    )
    llama_model: str = field(
        default_factory=lambda: _toml_val(
            _toml, "llama", "model", default="local-model"
        )
    )

    # ── Embeddings ─────────────────────────────────────────────
    embedding_provider: str = field(
        default_factory=lambda: _toml_val(
            _toml, "embeddings", "provider", default=""
        )
    )
    embedding_model: str = field(
        default_factory=lambda: _toml_val(
            _toml, "embeddings", "model", default=""
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: _toml_val(
            _toml, "embeddings", "base_url", default="http://localhost:8080/v1"
        )
    )

    # ── Web Search ─────────────────────────────────────────────
    search_provider: str = field(
        default_factory=lambda: _toml_val(
            _toml, "search", "provider", default="tavily"
        )
    )
    tavily_api_key: str = field(default_factory=lambda: _ev("TAVILY_API_KEY", ""))
    max_search_results_per_query: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "search", "max_results", default="10")
        )
    )
    top_urls_to_fetch: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "search", "top_urls_to_fetch", default="5")
        )
    )

    # ── Claims ─────────────────────────────────────────────────
    disable_claims: bool = field(
        default_factory=lambda: bool(
            _toml_val(_toml, "claims", "disable", default=False)
        )
    )

    # ── Performance / Limits ───────────────────────────────────
    state_timeout: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "state_timeout", default="90")
        )
    )
    page_fetch_timeout: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "page_fetch_timeout", default="10")
        )
    )
    max_chunks_per_page: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "max_chunks_per_page", default="3")
        )
    )
    max_contradiction_pairs: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "max_contradiction_pairs", default="5")
        )
    )
    max_tool_calls: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "max_tool_calls", default="40")
        )
    )
    min_sub_questions: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "min_sub_questions", default="3")
        )
    )
    max_sub_questions: int = field(
        default_factory=lambda: int(
            _toml_val(_toml, "performance", "max_sub_questions", default="5")
        )
    )

    # ── Agent ──────────────────────────────────────────────────
    session_id: str = field(
        default_factory=lambda: _toml_val(
            _toml, "agent", "session_id", default="default"
        )
    )

    # ── Logging ────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: _toml_val(
            _toml, "logging", "level", default="INFO"
        )
    )
    log_file: Optional[str] = field(
        default_factory=lambda: _toml_val(
            _toml, "logging", "file", default="research_agent.log"
        )
    )
    _verbosity_str: str = field(
        default_factory=lambda: _toml_val(
            _toml, "logging", "verbosity", default="normal"
        )
    )

    # ── Derived / compat ───────────────────────────────────────
    verbose_debug: bool = False  # set by main.py based on verbosity
    _vlevel: int = 1  # 0=minimal, 1=normal, 2=debug, 3=trace

    def __post_init__(self) -> None:
        vs = self._verbosity_str.lower()
        self._vlevel = _VERBOSITY_MAP.get(vs, 1)
        self.verbose_debug = self._vlevel >= 2

    @property
    def verbosity_level(self) -> int:
        return self._vlevel

    @verbosity_level.setter
    def verbosity_level(self, level: int) -> None:
        self._vlevel = max(0, min(3, level))
        self.verbose_debug = self._vlevel >= 2

    def validate(self) -> None:
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in .env or use LLM_PROVIDER=llamacpp for local models."
            )
        if self.search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily. "
                "Get a key at https://tavily.com"
            )


settings = Settings()
