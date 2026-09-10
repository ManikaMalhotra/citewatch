"""
Abstract base for LLM capture engines.

Defines the interface that all provider-specific capture implementations
must follow, plus shared retry/caching logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog

from citewatch.cache import CacheManager, make_cache_key
from citewatch.models import LLMResponse

logger = structlog.get_logger()


class BaseCaptureEngine(ABC):
    """Abstract base class for LLM capture engines."""

    provider: str = "unknown"

    def __init__(
        self,
        cache: CacheManager | None = None,
        max_concurrent: int = 3,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 2.0,
    ):
        self._cache = cache
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._retry_max = retry_max_attempts
        self._retry_delay = retry_base_delay

    @abstractmethod
    async def _call_api(self, query: str, model_id: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Provider-specific API call. Must be implemented by subclasses."""
        ...

    async def capture(
        self,
        query: str,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        variant_category: str = "original",
    ) -> LLMResponse:
        """
        Capture an LLM response with caching, retry, and concurrency control.
        """
        cache_key = make_cache_key(query, model_id, "api", temperature)

        # Check cache first
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.info("capture_cache_hit", model=model_id, key=cache_key[:12])
                return LLMResponse(**cached)

        # Acquire semaphore for concurrency control
        async with self._semaphore:
            response = await self._retry_call(query, model_id, temperature, max_tokens)

        response.cache_key = cache_key
        response.variant_category = variant_category

        # Store in cache
        if self._cache:
            self._cache.put(cache_key, response.model_dump(), query_text=query)

        return response

    async def _retry_call(
        self,
        query: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Retry API call with exponential backoff."""
        last_error = None
        for attempt in range(1, self._retry_max + 1):
            try:
                logger.info("capture_attempt", model=model_id, attempt=attempt, query=query[:60])
                return await self._call_api(query, model_id, temperature, max_tokens)
            except Exception as e:
                last_error = e
                if attempt < self._retry_max:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "capture_retry",
                        model=model_id,
                        attempt=attempt,
                        delay=delay,
                        error=str(e)[:100],
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Capture failed after {self._retry_max} attempts for model={model_id}: {last_error}"
        ) from last_error
