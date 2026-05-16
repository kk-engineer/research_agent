from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from src.config import settings


def extract_meaningful_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    sections: list[str] = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"]):
        if not isinstance(tag, Tag):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > settings.min_fragment_length:
            sections.append(text)

    return "\n\n".join(sections)


def split_into_chunks(text: str, max_chunk_size: int | None = None) -> list[str]:
    if max_chunk_size is None:
        max_chunk_size = settings.default_chunk_size
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [""]


def extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else url


def extract_relevant_sentences(
    text: str,
    query: str,
    max_sentences: int | None = None,
    min_sentence_len: int | None = None,
    max_sentence_len: int | None = None,
) -> list[str]:
    if max_sentences is None:
        max_sentences = settings.max_sentences
    if min_sentence_len is None:
        min_sentence_len = settings.min_sentence_length
    if max_sentence_len is None:
        max_sentence_len = settings.max_sentence_length

    query_words = {
        w.lower() for w in re.findall(r"[a-zA-Z]\w+", query) if len(w) > 2
    }

    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored: list[tuple[str, float]] = []

    bonus_keywords = tuple(settings.relevance_bonus_keywords)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < min_sentence_len or len(sent) > max_sentence_len:
            continue

        sent_lower = sent.lower()
        sent_words = {w for w in re.findall(r"[a-zA-Z]\w+", sent_lower) if len(w) > 2}
        if not sent_words:
            continue

        overlap = len(query_words & sent_words)

        bonus = settings.relevance_bonus_score if any(kw in sent_lower for kw in bonus_keywords) else 0.0

        score = overlap / max(len(query_words), 1) + bonus
        if overlap >= settings.sentence_min_overlap:
            scored.append((sent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    deduped: list[str] = []
    for sent, _ in scored:
        key = sent.lower()[: settings.dedup_prefix_length]
        if key not in seen:
            seen.add(key)
            deduped.append(sent)

    return deduped[:max_sentences]
