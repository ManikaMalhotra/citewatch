"""
AI-generated content detection for Citewatch.

Uses heuristic signals (entity density, sentence variance,
vocabulary richness) to flag potentially AI-generated content
that could cause deranking. No heavy ML dependencies (no GPT-2).
"""

from __future__ import annotations

import re
import structlog

from citewatch.models import AIContentScore

logger = structlog.get_logger()

# Common named entity patterns (simplified NER without ML)
ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",          # Multi-word proper nouns
    r"\b(?:https?://\S+)\b",                          # URLs
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",                   # Dates
    r"\b[A-Z]{2,}\b",                                  # Acronyms
    r"\b(?:v\d+\.\d+(?:\.\d+)?)\b",                   # Version numbers
]


def score_content(url: str, text: str) -> AIContentScore:
    """
    Score a page's content for AI-generation likelihood.

    Uses three heuristic signals:
    1. Entity density — AI text tends to be entity-sparse
    2. Sentence length variance — AI text has more uniform sentence lengths
    3. Vocabulary richness (type-token ratio) — AI text reuses words more
    """
    if not text or len(text) < 100:
        return AIContentScore(
            url=url,
            entity_density=0.0,
            sentence_length_variance=0.0,
            vocabulary_richness=0.0,
            overall_score=0.0,
            verdict="human",
            signals=[],
        )

    sentences = _split_sentences(text)
    words = text.lower().split()
    signals = []

    # Signal 1: Entity density
    entity_count = sum(len(re.findall(p, text)) for p in ENTITY_PATTERNS)
    entity_density = entity_count / max(len(sentences), 1)

    # Signal 2: Sentence length variance
    if len(sentences) > 2:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
        # Normalize: human text variance is typically 8-15, AI is 3-7
        sent_variance = variance
    else:
        sent_variance = 10.0  # Default to neutral

    # Signal 3: Vocabulary richness (type-token ratio)
    unique_words = set(words)
    ttr = len(unique_words) / max(len(words), 1)

    # Score each signal
    score = 0.0
    weight_total = 0.0

    # Low entity density → more likely AI
    if entity_density < 0.03:
        score += 0.4
        signals.append(f"Low entity density: {entity_density:.3f} (threshold: 0.03)")
    elif entity_density < 0.05:
        score += 0.2
        signals.append(f"Moderate entity density: {entity_density:.3f}")
    weight_total += 0.4

    # Low sentence variance → more likely AI
    if sent_variance < 4.0:
        score += 0.3
        signals.append(f"Low sentence variance: {sent_variance:.1f} (threshold: 4.0)")
    elif sent_variance < 6.0:
        score += 0.15
        signals.append(f"Moderate sentence variance: {sent_variance:.1f}")
    weight_total += 0.3

    # Low vocabulary richness → more likely AI
    if ttr < 0.35:
        score += 0.3
        signals.append(f"Low vocabulary richness: {ttr:.3f} (threshold: 0.35)")
    elif ttr < 0.45:
        score += 0.15
        signals.append(f"Moderate vocabulary richness: {ttr:.3f}")
    weight_total += 0.3

    # Normalize score
    overall = min(score / max(weight_total, 0.01), 1.0)

    # Verdict
    if overall > 0.7:
        verdict = "likely_ai"
    elif overall > 0.4:
        verdict = "mixed"
    else:
        verdict = "human"

    return AIContentScore(
        url=url,
        entity_density=entity_density,
        sentence_length_variance=sent_variance,
        vocabulary_richness=ttr,
        overall_score=overall,
        verdict=verdict,
        signals=signals,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic rules."""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 10]
