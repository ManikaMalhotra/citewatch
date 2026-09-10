"""
Google (Gemini) capture engine.

Captures LLM responses from Gemini models via the Google GenAI API,
using the same GEO-optimized capture strategy as the Anthropic engine.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from citewatch.capture.base import BaseCaptureEngine
from citewatch.capture.anthropic import (
    _capture_system_prompt,
    _extract_brand_mentions,
    _extract_citations,
    _extract_cot_steps,
    _extract_source_urls,
)
from citewatch.models import LLMResponse

logger = structlog.get_logger()


class GoogleCaptureEngine(BaseCaptureEngine):
    """Captures structured LLM responses from Gemini models."""

    provider = "google"

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
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=self.api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        messages = [
            SystemMessage(content=_capture_system_prompt()),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        raw_text = response.content

        # Token usage from Gemini
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
            }

        return LLMResponse(
            model=model_id,
            provider="google",
            query=query,
            mode="api",
            raw_text=raw_text,
            cot_steps=_extract_cot_steps(raw_text),
            citations=_extract_citations(raw_text),
            mentioned_brands=_extract_brand_mentions(raw_text),
            sources=_extract_source_urls(raw_text),
            timestamp=datetime.utcnow(),
            token_usage=usage,
        )
