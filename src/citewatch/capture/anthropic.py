"""
Anthropic (Claude) capture engine.

Captures LLM responses from Claude models via the Anthropic API,
using a carefully crafted system prompt that forces chain-of-thought
reasoning, citation extraction, and brand mention detection.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from citewatch.capture.base import BaseCaptureEngine
from citewatch.config import settings
from citewatch.models import BrandMention, Citation, CoTStep, LLMResponse

logger = structlog.get_logger()

def _capture_system_prompt() -> str:
    cfg = settings()
    context = cfg.target.domain_context or cfg.target.description or "the relevant product domain"
    return f"""You are a helpful technical assistant specializing in {context}.

When answering questions:

1. **Think step by step**: Break down your reasoning into clear numbered steps. Prefix each step with "Step N:".

2. **Cite sources**: When you reference a specific tool, library, or resource, mention where you learned about it (official docs, blog posts, etc.). Format citations as [Source: URL or description].

3. **Mention specific tools by name**: When discussing solutions, explicitly name the tools you're considering and explain why you recommend them. Cover both commercial and open-source options where relevant.

4. **Be specific about trade-offs**: Don't just recommend one tool — explain the pros/cons of alternatives.

5. **Include setup guidance**: When recommending a solution, include concrete steps or code snippets.

Your goal is to give the most helpful, comprehensive answer that a person searching for this information would want to see."""


GEO_CAPTURE_SYSTEM_PROMPT = _capture_system_prompt


class AnthropicCaptureEngine(BaseCaptureEngine):
    """Captures structured LLM responses from Claude models."""

    provider = "anthropic"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    async def _call_api(
        self,
        query: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        llm = ChatAnthropic(
            model=model_id,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        messages = [
            SystemMessage(content=_capture_system_prompt()),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        raw_text = response.content

        # Extract token usage from response metadata
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
            }

        # Parse the response into structured components
        cot_steps = _extract_cot_steps(raw_text)
        citations = _extract_citations(raw_text)
        brands = _extract_brand_mentions(raw_text)
        sources = _extract_source_urls(raw_text)

        return LLMResponse(
            model=model_id,
            provider="anthropic",
            query=query,
            mode="api",
            raw_text=raw_text,
            cot_steps=cot_steps,
            citations=citations,
            mentioned_brands=brands,
            sources=sources,
            timestamp=datetime.utcnow(),
            token_usage=usage,
        )


# ── Response Parsing Helpers ──────────────────────────────────────────────

def _known_brands() -> set[str]:
    """Brand names to scan for, from config (target + competitors + extras)."""
    try:
        names = set(settings().get_detectable_brands())
        if names:
            return names
    except Exception:
        pass
    return set()


def _extract_cot_steps(text: str) -> list[CoTStep]:
    """Extract numbered reasoning steps from the response."""
    steps = []
    import re
    # Match patterns like "Step 1:", "1.", "1)", "**Step 1:**"
    pattern = re.compile(r"(?:\*\*)?(?:Step\s+)?(\d+)[.):]\*?\*?\s*(.*?)(?=(?:\*\*)?(?:Step\s+)?\d+[.):]\*?\*?\s|$)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    for num_str, content in matches:
        content = content.strip()
        if len(content) > 10:  # Skip very short fragments
            steps.append(CoTStep(step=int(num_str), thinking=content[:2000]))
    return steps[:20]  # Cap at 20 steps


def _extract_citations(text: str) -> list[Citation]:
    """Extract citations/source references from the response."""
    import re
    citations = []
    seen = set()

    # Pattern 1: [Source: URL or description]
    for match in re.finditer(r"\[Source:\s*(.+?)\]", text, re.IGNORECASE):
        ref = match.group(1).strip()
        if ref not in seen:
            seen.add(ref)
            url = ref if ref.startswith("http") else ""
            citations.append(Citation(url=url, title=ref, snippet=""))

    # Pattern 2: Markdown links [text](url)
    for match in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if url not in seen:
            seen.add(url)
            citations.append(Citation(url=url, title=title, snippet=""))

    # Pattern 3: Bare URLs
    for match in re.finditer(r"(?<!\()(https?://[^\s\)>\]]+)", text):
        url = match.group(0).strip()
        if url not in seen:
            seen.add(url)
            citations.append(Citation(url=url, title="", snippet=""))

    return citations


def _extract_brand_mentions(text: str) -> list[BrandMention]:
    """Extract brand mentions from the response text."""
    import re
    brands = []
    known = _known_brands()
    # Longest names first so multi-word brands match before shorter aliases
    for brand in sorted(known, key=len, reverse=True):
        # Find all occurrences
        pattern = re.compile(re.escape(brand), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        # Get context around first mention (±100 chars)
        first = matches[0]
        start = max(0, first.start() - 100)
        end = min(len(text), first.end() + 100)
        context = text[start:end].strip()

        # Simple sentiment heuristic
        sentiment = "neutral"
        context_lower = context.lower()
        pos_signals = ["recommend", "best", "great", "excellent", "powerful", "popular", "leading"]
        neg_signals = ["expensive", "complex", "steep learning", "overkill", "vendor lock"]
        if any(s in context_lower for s in pos_signals):
            sentiment = "positive"
        elif any(s in context_lower for s in neg_signals):
            sentiment = "negative"

        # Check if explicitly recommended
        is_recommended = any(
            phrase in context_lower
            for phrase in ["i recommend", "i suggest", "consider using", "go with", "best option", "top choice"]
        )

        brands.append(BrandMention(
            brand=brand,
            context=context[:500],
            sentiment=sentiment,
            is_recommended=is_recommended,
            reason=f"Mentioned {len(matches)} time(s) in response",
        ))

    return brands


def _extract_source_urls(text: str) -> list[str]:
    """Extract all URLs from the response."""
    import re
    urls = re.findall(r"https?://[^\s\)>\]\"']+", text)
    return list(dict.fromkeys(urls))  # dedupe preserving order
