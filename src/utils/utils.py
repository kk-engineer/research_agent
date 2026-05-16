from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


def safe_get(data: Optional[dict], key: str, default: Any = None) -> Any:
    if data is None:
        return default
    return data.get(key, default)


async def with_timeout(coro, timeout_sec: float, label: str = "operation") -> Any:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning("⏱ Timeout after %.1fs: %s", timeout_sec, label)
        return None
    except Exception as e:
        logger.warning("Error in %s: %s", label, e)
        return None


def repair_json(raw: str) -> Optional[str]:
    if not raw:
        return None
    text = raw.strip()
    stripped = _strip_code_fences(text)
    if stripped != text:
        text = stripped
    if _is_valid_json(text):
        return text
    text = _fix_trailing_commas(text)
    if _is_valid_json(text):
        return text
    text = _fix_single_quotes(text)
    if _is_valid_json(text):
        return text
    text = _fix_unquoted_keys(text)
    if _is_valid_json(text):
        return text
    text = _truncate_after_json_end(text)
    if text and _is_valid_json(text):
        return text
    return None


def parse_json_safely(
    raw: str, model_name: str = "response"
) -> Optional[dict[str, Any]]:
    if not raw:
        logger.warning("Empty LLM response when parsing %s", model_name)
        return None
    document = raw.strip()
    stripped = _strip_code_fences(document)
    if stripped != document:
        document = stripped
    result = _try_parse(document)
    if result is not None:
        return result
    result = _try_parse(_fix_trailing_commas(document))
    if result is not None:
        return result
    result = _try_parse(_fix_single_quotes(document))
    if result is not None:
        return result
    result = _try_parse(_fix_unquoted_keys(document))
    if result is not None:
        return result
    extracted = extract_json_from_llm_response(document)
    if extracted is not None:
        logger.info("Recovered JSON via regex extraction for %s", model_name)
        return extracted
    truncated = _truncate_after_json_end(document)
    if truncated:
        result = _try_parse(truncated)
        if result is not None:
            logger.info("Recovered JSON via truncation for %s", model_name)
            return result
    logger.warning("All JSON repair strategies failed for %s", model_name)
    logger.debug("Raw response (first 500): %s", document[:500])
    return None


def _try_parse(text: Optional[str]) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*\n\s*}", "}", text)
    text = re.sub(r",\s*\n\s*]", "]", text)
    return text


def _fix_single_quotes(text: str) -> str:
    result = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == "'" and not in_string:
            result.append('"')
            in_string = True
            continue
        if ch == "'" and in_string:
            result.append('"')
            in_string = False
            continue
        if ch == '"' and in_string:
            in_string = False
            result.append(ch)
            continue
        if ch == '"' and not in_string:
            in_string = True
            result.append(ch)
            continue
        result.append(ch)
    return "".join(result)


def _fix_unquoted_keys(text: str) -> str:
    text = re.sub(
        r"([{,]\s*)(\w+)(\s*:)",
        lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
        text,
    )
    return text


def _truncate_after_json_end(text: str) -> Optional[str]:
    stack: list[str] = []
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                if not stack:
                    return text[: i + 1]
            else:
                return None
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                if not stack:
                    return text[: i + 1]
            else:
                return None
    return None
