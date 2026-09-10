"""
Competitor content crawling & analysis for Citewatch.

Discovers, crawls, and embeds competitor content for head-to-head
article comparison and keyword gap analysis.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import chromadb
import structlog

from citewatch.embeddings import EmbeddingManager, get_embedding_manager
from citewatch.ingestion.crawler import crawl_urls
from citewatch.ingestion.sitemap import fetch_sitemap_urls, filter_urls_by_depth
from citewatch.models import CompetitorPage, CrawledPage, KeywordCluster

logger = structlog.get_logger()


class CompetitorKB:
    """ChromaDB-backed knowledge base for competitor content (chunk-level)."""

    def __init__(self, db_path: str = "./.citewatch_competitor_kb", embedding_mgr: EmbeddingManager | None = None):
        from pathlib import Path
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._emb = embedding_mgr or get_embedding_manager()
        self._collection = self._client.get_or_create_collection(
            name="competitor_content",
            embedding_function=self._emb.chroma_ef,
        )
        logger.info("competitor_kb_initialized", path=str(self.db_path))

    def add_pages(self, pages: list[CompetitorPage]) -> None:
        """Store competitor page content with embeddings (page-level, for backwards compat)."""
        ids = []
        documents = []
        metadatas = []

        for page in pages:
            if not page.body_text or len(page.body_text) < 50:
                continue
            page_id = f"{page.domain}_{hash(page.url) & 0xFFFFFFFF:08x}"
            ids.append(page_id)
            doc_text = f"{page.title}\n\n{page.body_text[:2000]}"
            documents.append(doc_text)
            metadatas.append({
                "url": page.url,
                "domain": page.domain,
                "title": page.title,
                "word_count": page.word_count,
                "has_schema": str(page.has_schema_markup),
                "type": "page",
            })

        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("competitor_pages_stored", count=len(ids))

    def add_chunks(self, chunk_ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """Store chunked competitor content with embeddings (like the main KB)."""
        self._collection.upsert(ids=chunk_ids, documents=documents, metadatas=metadatas)
        logger.info("competitor_chunks_stored", count=len(chunk_ids))

    def find_similar(self, query: str, n_results: int = 5, domain: str | None = None) -> dict:
        """Find competitor pages most similar to a query/topic."""
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if domain:
            kwargs["where"] = {"domain": domain}
        return self._collection.query(**kwargs)

    def get_domains(self) -> list[str]:
        """Return all unique domains in the KB."""
        try:
            results = self._collection.get(include=["metadatas"], limit=10000)
            domains = set()
            if results and results.get("metadatas"):
                for meta in results["metadatas"]:
                    if meta and "domain" in meta:
                        domains.add(meta["domain"])
            return sorted(domains)
        except Exception:
            return []

    def get_domain_stats(self, domain: str) -> dict:
        """Get stats for a specific competitor domain."""
        try:
            results = self._collection.get(
                where={"domain": domain},
                include=["metadatas"],
                limit=10000,
            )
            if not results or not results.get("metadatas"):
                return {"domain": domain, "pages": 0, "chunks": 0, "total_words": 0, "schema_count": 0}

            metas = results["metadatas"]
            pages = set()
            chunks = 0
            total_words = 0
            schema_count = 0
            for meta in metas:
                if not meta:
                    continue
                url = meta.get("url", "")
                if url:
                    pages.add(url)
                chunks += 1
                total_words += int(meta.get("word_count", 0))
                if meta.get("has_schema") == "True":
                    schema_count += 1

            return {
                "domain": domain,
                "pages": len(pages),
                "chunks": chunks,
                "total_words": total_words,
                "avg_words_per_page": total_words // max(len(pages), 1),
                "schema_count": schema_count,
                "schema_pct": round(schema_count / max(len(pages), 1) * 100, 1),
            }
        except Exception as e:
            logger.warning("domain_stats_failed", domain=domain, error=str(e)[:100])
            return {"domain": domain, "pages": 0, "chunks": 0, "total_words": 0}

    @property
    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection("competitor_content")
        self._collection = self._client.get_or_create_collection(
            name="competitor_content", embedding_function=self._emb.chroma_ef,
        )



async def discover_competitor_urls(
    competitor_domains: list[str],
    keyword_clusters: list[KeywordCluster] | None = None,
    max_urls_per_domain: int = 100,
    max_depth: int = 3,
) -> dict[str, list[str]]:
    """
    Discover relevant competitor URLs from their sitemaps.

    If keyword_clusters are provided, filters to only URLs likely relevant
    to our keyword clusters (by checking URL path keywords).

    Returns:
        Dict mapping domain -> list of URLs to crawl.
    """
    domain_urls: dict[str, list[str]] = {}

    # Build keyword filter from clusters
    filter_keywords: set[str] = set()
    if keyword_clusters:
        for cluster in keyword_clusters:
            for kw in cluster.keywords:
                # Extract significant words (3+ chars)
                for word in kw.lower().split():
                    if len(word) >= 3 and word not in {"the", "and", "for", "how", "what", "with", "best", "top"}:
                        filter_keywords.add(word)

    for domain in competitor_domains:
        clean_domain = domain.strip().lower().replace("www.", "")
        sitemap_url = f"https://{clean_domain}/sitemap.xml"

        logger.info("discovering_competitor", domain=clean_domain, sitemap=sitemap_url)

        try:
            entries = await fetch_sitemap_urls(sitemap_url, max_depth=max_depth)
            entries = filter_urls_by_depth(entries, max_depth)
        except Exception as e:
            logger.warning("competitor_sitemap_failed", domain=clean_domain, error=str(e))
            continue

        urls = [e["url"] for e in entries]

        # Filter to relevant URLs if we have keyword clusters
        if filter_keywords:
            relevant_urls = []
            for url in urls:
                path = urlparse(url).path.lower()
                # Check if URL path contains any of our target keywords
                if any(kw in path for kw in filter_keywords):
                    relevant_urls.append(url)
            urls = relevant_urls or urls[:max_urls_per_domain]  # fallback to first N if no matches

        # Cap per domain
        urls = urls[:max_urls_per_domain]
        domain_urls[clean_domain] = urls

        logger.info("competitor_urls_discovered", domain=clean_domain, total=len(entries), filtered=len(urls))

    return domain_urls


async def crawl_competitor_content(
    domain_urls: dict[str, list[str]],
    max_concurrent: int = 3,
    request_delay: float = 1.0,
) -> list[CompetitorPage]:
    """
    Crawl competitor pages and convert to CompetitorPage objects.

    Uses the existing crawler with polite delays to avoid rate limiting.
    """
    all_pages: list[CompetitorPage] = []

    for domain, urls in domain_urls.items():
        if not urls:
            continue

        logger.info("crawling_competitor", domain=domain, urls=len(urls))

        # Crawl pages using existing infrastructure
        crawled: list[CrawledPage] = await crawl_urls(
            urls,
            max_concurrent=max_concurrent,
            request_delay=request_delay,
        )

        # Convert CrawledPage -> CompetitorPage
        for page in crawled:
            if page.status_code != 200 or not page.body_text:
                continue

            comp_page = CompetitorPage(
                url=page.url,
                domain=domain,
                title=page.title,
                meta_description=page.meta_description,
                body_text=page.body_text,
                word_count=page.word_count,
                headings=page.headings,
                has_schema_markup=page.has_schema_markup,
                crawled_at=page.crawled_at,
            )
            all_pages.append(comp_page)

    logger.info("competitor_crawl_complete", total_pages=len(all_pages))
    return all_pages


def match_articles(
    our_pages: list[CrawledPage],
    competitor_pages: list[CompetitorPage],
    clusters: list[KeywordCluster],
    similarity_threshold: float = 0.60,
) -> list[tuple[CrawledPage, CompetitorPage, float, str]]:
    """
    Match our pages to competitor pages by embedding similarity.

    Returns list of (our_page, competitor_page, similarity, cluster_keyword) tuples.
    """
    if not our_pages or not competitor_pages:
        return []

    emb = get_embedding_manager()

    # Embed our page titles
    our_texts = [f"{p.title} {p.meta_description}" for p in our_pages]
    our_embeddings = emb.embed_texts(our_texts)

    # Embed competitor page titles
    comp_texts = [f"{p.title} {p.meta_description}" for p in competitor_pages]
    comp_embeddings = emb.embed_texts(comp_texts)

    matches: list[tuple[CrawledPage, CompetitorPage, float, str]] = []

    for i, our_page in enumerate(our_pages):
        best_sim = 0.0
        best_comp = None
        best_j = -1

        for j, comp_page in enumerate(competitor_pages):
            sim = emb.cosine_similarity(our_embeddings[i], comp_embeddings[j])
            if sim > best_sim:
                best_sim = sim
                best_comp = comp_page
                best_j = j

        if best_comp and best_sim > similarity_threshold:
            # Try to find which cluster this belongs to
            cluster_kw = ""
            our_url_lower = our_page.url.lower()
            for cluster in clusters:
                if any(our_url_lower == page_url.lower() for page_url in cluster.our_pages):
                    cluster_kw = cluster.primary_keyword
                    break

            matches.append((our_page, best_comp, best_sim, cluster_kw))

    # Sort by similarity descending
    matches.sort(key=lambda m: m[2], reverse=True)

    logger.info("article_matches_found", total=len(matches))
    return matches
