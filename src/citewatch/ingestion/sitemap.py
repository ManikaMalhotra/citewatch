"""
Sitemap parser for Citewatch.

Fetches and parses XML sitemaps, handling sitemap indexes
(nested sitemaps) recursively. Extracts URLs with lastmod timestamps.
"""

from __future__ import annotations

import structlog
import httpx
from bs4 import BeautifulSoup

from citewatch.ingestion.crawler import DEFAULT_USER_AGENT

logger = structlog.get_logger()


async def fetch_sitemap_urls(
    sitemap_url: str,
    max_depth: int = 4,
    _current_depth: int = 0,
) -> list[dict]:
    """
    Fetch and parse a sitemap, returning a list of URL entries.

    Handles sitemap index files by recursively fetching sub-sitemaps.

    Returns:
        List of dicts with keys: url, lastmod, changefreq, priority
    """
    if _current_depth > max_depth:
        logger.warning("sitemap_max_depth", url=sitemap_url, depth=_current_depth)
        return []

    logger.info("fetching_sitemap", url=sitemap_url, depth=_current_depth)

    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        follow_redirects=True,
    ) as client:
        try:
            resp = await client.get(sitemap_url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("sitemap_fetch_failed", url=sitemap_url, error=str(e))
            return []

    soup = BeautifulSoup(resp.content, "xml")
    entries = []

    # Check if this is a sitemap index (contains <sitemap> elements)
    sitemaps = soup.find_all("sitemap")
    if sitemaps:
        logger.info("sitemap_index_found", count=len(sitemaps))
        for sm in sitemaps:
            loc = sm.find("loc")
            if loc and loc.text:
                sub_entries = await fetch_sitemap_urls(
                    loc.text.strip(),
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1,
                )
                entries.extend(sub_entries)
        return entries

    # Regular sitemap: extract <url> elements
    urls = soup.find_all("url")
    for url_elem in urls:
        loc = url_elem.find("loc")
        if not loc or not loc.text:
            continue

        entry = {
            "url": loc.text.strip(),
            "lastmod": "",
            "changefreq": "",
            "priority": "",
        }

        lastmod = url_elem.find("lastmod")
        if lastmod and lastmod.text:
            entry["lastmod"] = lastmod.text.strip()

        changefreq = url_elem.find("changefreq")
        if changefreq and changefreq.text:
            entry["changefreq"] = changefreq.text.strip()

        priority = url_elem.find("priority")
        if priority and priority.text:
            entry["priority"] = priority.text.strip()

        entries.append(entry)

    logger.info("sitemap_parsed", url=sitemap_url, entries=len(entries))
    return entries


def filter_urls_by_domain(entries: list[dict], domain: str) -> list[dict]:
    """Filter sitemap entries to only include URLs from the target domain."""
    return [e for e in entries if domain in e["url"]]


def filter_urls_by_depth(entries: list[dict], max_depth: int) -> list[dict]:
    """Filter URLs by path depth (number of / segments after domain)."""
    filtered = []
    for entry in entries:
        url = entry["url"]
        # Count path segments after the domain
        try:
            from urllib.parse import urlparse
            path = urlparse(url).path.strip("/")
            depth = len(path.split("/")) if path else 0
            if depth <= max_depth:
                filtered.append(entry)
        except Exception:
            filtered.append(entry)  # Include on parse failure
    return filtered
