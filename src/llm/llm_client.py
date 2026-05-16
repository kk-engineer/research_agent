from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional
from openai import AsyncOpenAI



from src.config import settings
from src.monitoring.llm_logger import log_llm_call_start, log_llm_call_end
from src.utils.utils import parse_json_safely, with_timeout

logger = logging.getLogger(__name__)

_SHORT = 300


def _trunc(s: str, n: int = _SHORT) -> str:
    return s[:n] + "…" if len(s) > n else s


class LLMClient(ABC):
    def __init__(self) -> None:
        self.on_token: Optional[Callable[[str], None]] = None

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type | None = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> Any: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def check_connectivity(self) -> None:
        """Verify that the configured LLM and embedding endpoints are reachable.
        Raises ConnectionError with a descriptive message on failure."""

    async def close(self) -> None:
        """Close any underlying HTTP clients. Override in subclasses that hold resources."""


class _BaseOpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        embedding_model: str = "",
        embedding_base_url: str = "",
    ) -> None:
        super().__init__()
        self._model = model
        self._provider = "openai" if "openai" in str(base_url) or not base_url else "llamacpp"
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

        self._embedding_model = embedding_model or model
        if embedding_base_url and embedding_base_url != base_url:
            self._embed_client: Optional[AsyncOpenAI] = AsyncOpenAI(
                api_key="not-needed", base_url=embedding_base_url
            )
            logger.info(
                "Dedicated embedding client — base_url=%s model=%s",
                embedding_base_url,
                self._embedding_model,
            )
        else:
            self._embed_client = None

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type | None = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> Any:
        if max_tokens is None:
            max_tokens = settings.llm_max_tokens
        purpose = response_model.__name__ if response_model else "free_text"
        log_purpose = purpose.replace("Response", "").replace("Analysis", "")
        logger.info(
            "🤖 LLM call  [%s]  model=%s  max_tokens=%d  temp=%.1f",
            log_purpose,
            self._model,
            max_tokens,
            temperature,
        )
        logger.debug(
            "  SYSTEM: %s",
            _trunc(system_prompt.replace("\n", " "), 200),
        )
        logger.debug("  USER: %s", _trunc(user_prompt.replace("\n", " ")))

        llm_start = log_llm_call_start(
            model=self._model,
            provider=self._provider,
            purpose=log_purpose,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_model:
            kwargs["response_format"] = {"type": "json_object"}

        timeout = request_timeout or settings.llm_request_timeout
        start = time.monotonic()

        use_stream = self.on_token is not None

        if use_stream:
            kwargs["stream"] = True

        response = await with_timeout(
            self._client.chat.completions.create(**kwargs),
            timeout_sec=timeout,
            label=f"LLM {self._model}/{log_purpose}",
        )

        elapsed = time.monotonic() - start
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if response is None:
            logger.warning("  LLM ← [TIMEOUT/ERROR] after %.2fs", elapsed)
            log_llm_call_end(
                purpose=log_purpose,
                start=llm_start,
                success=False,
                model=self._model,
                provider=self._provider,
                temperature=temperature,
                max_tokens=max_tokens,
                response_content="",
            )
            if response_model:
                return _fallback_for_model(response_model)
            return ""

        if use_stream:
            content_parts: list[str] = []
            last_chunk = None
            async for chunk in response:
                last_chunk = chunk
                delta = chunk.choices[0].delta if chunk.choices else None
                token = (delta.content or "") if delta else ""
                if token:
                    content_parts.append(token)
                    self.on_token(token)
            content = "".join(content_parts)
            if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                u = last_chunk.usage
                prompt_tokens = u.prompt_tokens or 0
                completion_tokens = u.completion_tokens or 0
                total_tokens = u.total_tokens or 0
        else:
            content = response.choices[0].message.content or ""
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                prompt_tokens = u.prompt_tokens or 0
                completion_tokens = u.completion_tokens or 0
                total_tokens = u.total_tokens or 0

        elapsed = time.monotonic() - start

        usage_prefix = ""
        if prompt_tokens > 0:
            usage_prefix = f"  in={prompt_tokens}  out={completion_tokens}  total={total_tokens}"
        logger.info(
            "  LLM ← %d chars%s  [%s]  (%.2fs)",
            len(content),
            usage_prefix,
            log_purpose,
            elapsed,
        )
        logger.debug("  LLM ← %s", _trunc(content))

        log_llm_call_end(
            purpose=log_purpose,
            start=llm_start,
            response_content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=True,
            model=self._model,
            provider=self._provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response_model:
            model_name = response_model.__name__
            parsed = parse_json_safely(content, model_name=model_name)
            if parsed is not None:
                try:
                    return response_model(**parsed)
                except Exception:
                    defaults = _fallback_for_model(response_model)
                    if defaults is not None and hasattr(defaults, "model_dump"):
                        merged = {**defaults.model_dump(), **parsed}
                        try:
                            return response_model(**merged)
                        except Exception:
                            pass
            fallback = _fallback_for_model(response_model)
            return fallback

        return content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._embed_client or self._client
        start = time.monotonic()
        logger.info("🤖 Embedding %d text(s)  model=%s", len(texts), self._embedding_model)

        response = await with_timeout(
            client.embeddings.create(model=self._embedding_model, input=texts),
            timeout_sec=settings.llm_request_timeout,
            label="embed",
        )

        elapsed = time.monotonic() - start
        if response is None:
            logger.warning("  Embed ← [TIMEOUT] after %.2fs", elapsed)
            return [[0.0] * 384 for _ in texts]

        logger.info("  Embed ← %d vectors  (%.2fs)", len(response.data), elapsed)
        return [d.embedding for d in response.data]

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass
        if self._embed_client is not None:
            try:
                await self._embed_client.close()
            except Exception:
                pass

    async def check_connectivity(self) -> None:
        timeout = settings.llm_request_timeout
        llm_url = str(self._client.base_url)

        try:
            await asyncio.wait_for(self._client.models.list(), timeout=timeout)
            logger.info("LLM endpoint reachable: %s (model=%s)", llm_url, self._model)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"LLM endpoint timed out after {timeout}s: {llm_url}. "
                f"Check that your server is running."
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM endpoint at {llm_url}: {e}"
            )

        if self._embed_client is not None:
            embed_url = str(self._embed_client.base_url)
            try:
                await asyncio.wait_for(
                    self._embed_client.models.list(), timeout=timeout
                )
                logger.info(
                    "Embedding endpoint reachable: %s (model=%s)",
                    embed_url,
                    self._embedding_model,
                )
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"Embedding endpoint timed out after {timeout}s: {embed_url}. "
                    f"Check that your embedding server is running."
                )
            except Exception as e:
                raise ConnectionError(
                    f"Cannot reach embedding endpoint at {embed_url}: {e}"
                )


class OpenAIClient(_BaseOpenAIClient):
    def __init__(self) -> None:
        super().__init__(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            embedding_base_url=settings.embedding_base_url,
        )


class LlamacppClient(_BaseOpenAIClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="not-needed",
            base_url=settings.llama_base_url,
            model=settings.llama_model,
            embedding_model=settings.embedding_model or settings.llama_model,
            embedding_base_url=settings.embedding_base_url or settings.llama_base_url,
        )
        self._provider = "llamacpp"
        logger.info(
            "LlamacppClient initialised — base_url=%s model=%s",
            settings.llama_base_url,
            settings.llama_model,
        )


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type | None = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> Any:
        if response_model is not None:
            return response_model(
                primary_intent="research",
                scope="general",
                ambiguities=[],
                has_ambiguities=False,
            )
        return "Mock response"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

    async def check_connectivity(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _fallback_for_model(model: type) -> Any:
    from src import models as m
    from src.core.sub_query_planner import DecompositionResponse

    registry = {
        m.QueryAnalysis: lambda: m.QueryAnalysis(
            original_query="",
            primary_intent="unknown",
            scope="broad",
            ambiguities=[],
            has_ambiguities=False,
        ),
        m.ClarificationResponse: lambda: m.ClarificationResponse(
            question="Could you please clarify your question?",
            resolved_query="",
            needs_clarification=False,
        ),
        m.SubQuestion: lambda: m.SubQuestion(text=""),
        DecompositionResponse: lambda: DecompositionResponse(
            subqueries=[]
        ),
        m.SynthesisResponse: lambda: m.SynthesisResponse(),
    }
    fallback_fn = registry.get(model)
    if fallback_fn:
        return fallback_fn()
    logger.warning("No fallback registered for %s", model.__name__)
    try:
        return model()
    except Exception:
        return {}


def get_llm_client() -> LLMClient:
    provider = settings.llm_provider
    if settings.embedding_base_url:
        logger.info(
            "Embedding: model=%s  base_url=%s",
            settings.embedding_model or settings.llm_model,
            settings.embedding_base_url,
        )
    if provider == "mock":
        logger.info("Using MockLLMClient")
        return MockLLMClient()
    if provider == "llamacpp":
        logger.info("Using LlamacppClient")
        return LlamacppClient()
    logger.info("Using OpenAIClient — model=%s", settings.llm_model)
    return OpenAIClient()
