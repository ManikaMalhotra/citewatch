"""
HTTP crawler for Citewatch.

Async crawler using httpx + BeautifulSoup for HTML parsing.
Supports incremental crawling via etag/last-modified tracking.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
import structlog
from bs4 import BeautifulSoup

from citewatch.models import CrawledPage

logger = structlog.get_logger()

DEFAULT_USER_AGENT = "citewatch/0.1.0 (+https://github.com/ManikaMalhotra/citewatch)"

# Tags to strip from HTML before text extraction
STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe"}

async def crawl_urls(
    urls: list[str],
    max_concurrent: int = 5,
    request_delay: float = 0.5,
    existing_pages: dict[str, CrawledPage] | None = None,
    user_agent: str | None = None,
) -> list[CrawledPage]:
    """
    Crawl a list of URLs and return parsed page data.

    Args:
        urls: URLs to crawl.
        max_concurrent: Max concurrent HTTP requests.
        request_delay: Delay between requests (polite crawling).
        existing_pages: Previously crawled pages for incremental mode.

    Returns:
        List of CrawledPage objects.
    """
    existing = existing_pages or {}
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[CrawledPage] = []

    async with httpx.AsyncClient(
        timeout=30.0,
        headers={
            "User-Agent": user_agent or DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
        follow_redirects=True,
    ) as client:
        tasks = []
        for url in urls:
            tasks.append(_crawl_single(client, url, semaphore, request_delay, existing.get(url)))

        pages = await asyncio.gather(*tasks, return_exceptions=True)

    for page in pages:
        if isinstance(page, Exception):
            logger.error("crawl_error", error=str(page))
        elif page is not None:
            results.append(page)

    logger.info("crawl_complete", total=len(urls), success=len(results))
    return results


async def _crawl_single(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    delay: float,
    existing: CrawledPage | None,
) -> CrawledPage | None:
    """Crawl a single URL with incremental support."""
    async with semaphore:
        try:
            # Build conditional headers for incremental crawling
            headers = {}
            if existing and existing.etag:
                headers["If-None-Match"] = existing.etag
            if existing and existing.last_modified:
                headers["If-Modified-Since"] = existing.last_modified

            resp = await client.get(url, headers=headers)

            # 304 Not Modified — page hasn't changed
            if resp.status_code == 304:
                logger.debug("page_not_modified", url=url)
                return existing

            resp.raise_for_status()

            # Polite delay
            await asyncio.sleep(delay)

            # Parse the page
            page = _parse_html(url, resp)
            return page

        except httpx.HTTPError as e:
            logger.warning("page_fetch_failed", url=url, error=str(e))
            return CrawledPage(url=url, status_code=getattr(e, "response", None) and e.response.status_code or 0)
        except Exception as e:
            logger.error("page_parse_error", url=url, error=str(e))
            return None


def _parse_html(url: str, response: httpx.Response) -> CrawledPage:
    """Parse an HTTP response into a CrawledPage."""
    soup = BeautifulSoup(response.content, "lxml")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta_desc = meta_tag.get("content", "")

    # Extract headings
    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            text = h.get_text(strip=True)
            if text:
                headings.append(f"h{level}: {text}")

    # Strip non-content tags
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Extract body text
    body = soup.find("body")
    body_text = ""
    if body:
        body_text = body.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        body_text = re.sub(r"\n{3,}", "\n\n", body_text)
        body_text = re.sub(r" {2,}", " ", body_text)

    # Detect schema markup
    schema_types = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "")
            if isinstance(data, dict) and "@type" in data:
                schema_types.append(data["@type"])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "@type" in item:
                        schema_types.append(item["@type"])
        except (json.JSONDecodeError, TypeError):
            pass

    return CrawledPage(
        url=url,
        title=title,
        meta_description=meta_desc,
        headings=headings,
        body_text=body_text,
        last_modified=response.headers.get("Last-Modified", ""),
        etag=response.headers.get("ETag", ""),
        status_code=response.status_code,
        crawled_at=datetime.utcnow(),
        word_count=len(body_text.split()),
        has_schema_markup=len(schema_types) > 0,
        schema_types=schema_types,
    )
