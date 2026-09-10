"""
Semantic text chunker for Citewatch.

Heading-aware chunking that preserves section hierarchy.
Chunks by approximate token count with configurable overlap.
"""

from __future__ import annotations

import hashlib
import re

import structlog

from citewatch.models import ContentChunk

logger = structlog.get_logger()

# Rough approximation: 1 token ≈ 4 characters (for English text)
CHARS_PER_TOKEN = 4


def chunk_page(
    url: str,
    text: str,
    headings: list[str] | None = None,
    chunk_size: int = 512,
    overlap: float = 0.2,
) -> list[ContentChunk]:
    """
    Split page text into semantic chunks, respecting heading boundaries.

    Args:
        url: Source URL for metadata.
        text: Full page body text.
        headings: List of headings for context.
        chunk_size: Target chunk size in tokens.
        overlap: Fraction of overlap between chunks (0.0–0.5).

    Returns:
        List of ContentChunk objects.
    """
    if not text or not text.strip():
        return []

    target_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = int(target_chars * overlap)

    # Split text into sections by headings
    sections = _split_by_headings(text)

    chunks = []
    position = 0

    for heading, section_text in sections:
        if not section_text.strip():
            continue

        # If section is small enough, it's one chunk
        if len(section_text) <= target_chars:
            chunk_id = _make_chunk_id(url, position)
            chunks.append(ContentChunk(
                chunk_id=chunk_id,
                source_url=url,
                text=section_text.strip(),
                heading_context=heading,
                position=position,
                token_count=len(section_text) // CHARS_PER_TOKEN,
                metadata={"url": url, "heading": heading, "position": position},
            ))
            position += 1
            continue

        # Split large sections into overlapping chunks
        start = 0
        while start < len(section_text):
            end = start + target_chars

            # Try to break at a sentence boundary
            if end < len(section_text):
                # Look for sentence-ending punctuation near the target
                boundary = _find_sentence_boundary(section_text, end - 100, end + 100)
                if boundary > start:
                    end = boundary

            chunk_text = section_text[start:end].strip()
            if chunk_text:
                chunk_id = _make_chunk_id(url, position)
                chunks.append(ContentChunk(
                    chunk_id=chunk_id,
                    source_url=url,
                    text=chunk_text,
                    heading_context=heading,
                    position=position,
                    token_count=len(chunk_text) // CHARS_PER_TOKEN,
                    metadata={"url": url, "heading": heading, "position": position},
                ))
                position += 1

            # Move forward, accounting for overlap
            start = end - overlap_chars
            if start >= len(section_text):
                break

    logger.debug("page_chunked", url=url, chunks=len(chunks))
    return chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, content) pairs based on heading patterns."""
    # Match common heading patterns in extracted text
    heading_pattern = re.compile(r"^(#{1,6}\s+.+|h[1-6]:\s*.+)$", re.MULTILINE)

    parts = heading_pattern.split(text)
    sections = []
    current_heading = ""

    if parts and not heading_pattern.match(parts[0]):
        # Text before first heading
        sections.append(("", parts[0]))
        parts = parts[1:]

    i = 0
    while i < len(parts):
        if heading_pattern.match(parts[i]):
            current_heading = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((current_heading, content))
            i += 2
        else:
            sections.append((current_heading, parts[i]))
            i += 1

    if not sections:
        sections = [("", text)]

    return sections


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find the best sentence boundary within a range."""
    start = max(0, start)
    end = min(len(text), end)
    segment = text[start:end]

    # Look for sentence-ending punctuation followed by whitespace
    for pattern in [r"\.\s", r"\!\s", r"\?\s", r"\n\n"]:
        matches = list(re.finditer(pattern, segment))
        if matches:
            # Use the last match (closest to target)
            return start + matches[-1].end()

    return end  # No good boundary found, use target


def _make_chunk_id(url: str, position: int) -> str:
    """Generate a unique chunk ID from URL + position."""
    raw = f"{url}|{position}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]
