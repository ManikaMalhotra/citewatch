"""
SEO Audit module for Citewatch.

Analyzes GSC/Ahrefs CSV exports for:
- Keyword cannibalization (pairwise cosine similarity)
- On-page gaps (missing schema, duplicate titles)
- E-E-A-T gap analysis vs. competitors
"""

from __future__ import annotations

import csv
from pathlib import Path

import structlog

from citewatch.audit.ai_detect import score_content
from citewatch.cache import KnowledgeBase
from citewatch.embeddings import get_embedding_manager
from citewatch.models import (
    AIContentScore,
    CannibalizationCluster,
    CrawledPage,
    FixPriority,
    RecommendedFix,
    SEOAuditResult,
)

logger = structlog.get_logger()


async def run_seo_audit(
    pages: list[CrawledPage],
    kb: KnowledgeBase,
    gsc_csv: str | None = None,
    ahrefs_csv: str | None = None,
    cannibalization_threshold: float = 0.82,
) -> SEOAuditResult:
    """
    Run a comprehensive SEO audit.

    Args:
        pages: Crawled page data from ingestion.
        kb: Knowledge base for embeddings.
        gsc_csv: Path to GSC CSV export.
        ahrefs_csv: Path to Ahrefs CSV export.
        cannibalization_threshold: Cosine sim threshold for cannibalization.
    """
    logger.info("seo_audit_start", pages=len(pages))

    result = SEOAuditResult(total_pages_analyzed=len(pages))

    # 1. Cannibalization detection
    if pages:
        result.cannibalization_clusters = await _detect_cannibalization(
            pages, cannibalization_threshold
        )

    # 2. AI content scoring
    result.ai_content_scores = _score_all_pages(pages)

    # 3. Missing schema detection
    result.missing_schema_pages = [
        p.url for p in pages
        if not p.has_schema_markup and p.word_count > 200
    ]

    # 4. Duplicate titles
    result.duplicate_titles = _find_duplicate_titles(pages)

    # 5. GSC analysis (if available)
    gsc_data = _load_gsc_csv(gsc_csv) if gsc_csv else []

    # 6. Ahrefs analysis (if available)
    ahrefs_data = _load_ahrefs_csv(ahrefs_csv) if ahrefs_csv else []

    # 7. Generate top recommendations
    result.top_recommendations = _generate_recommendations(result, gsc_data, ahrefs_data)

    logger.info(
        "seo_audit_complete",
        clusters=len(result.cannibalization_clusters),
        ai_flags=len([s for s in result.ai_content_scores if s.verdict != "human"]),
        missing_schema=len(result.missing_schema_pages),
    )

    return result


async def _detect_cannibalization(
    pages: list[CrawledPage],
    threshold: float,
) -> list[CannibalizationCluster]:
    """Detect pages competing for similar queries via title embedding similarity."""
    emb = get_embedding_manager()

    # Embed page titles
    titles = [p.title or p.url for p in pages]
    if len(titles) < 2:
        return []

    logger.info("computing_cannibalization", pages=len(titles))
    embeddings = emb.embed_texts(titles)

    # Pairwise similarity (O(n²) — fine for <5000 pages)
    clusters = []
    visited = set()

    for i in range(len(pages)):
        if i in visited:
            continue
        cluster_pages = [pages[i].url]
        cluster_queries = [titles[i]]
        max_sim = 0.0

        for j in range(i + 1, len(pages)):
            if j in visited:
                continue
            sim = emb.cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                cluster_pages.append(pages[j].url)
                cluster_queries.append(titles[j])
                max_sim = max(max_sim, sim)
                visited.add(j)

        if len(cluster_pages) > 1:
            visited.add(i)
            clusters.append(CannibalizationCluster(
                queries=cluster_queries,
                pages=cluster_pages,
                max_similarity=max_sim,
                recommendation=f"Consider consolidating {len(cluster_pages)} pages targeting similar topics",
            ))

    return clusters


def _score_all_pages(pages: list[CrawledPage]) -> list[AIContentScore]:
    """Run AI content detection on all pages."""
    scores = []
    for page in pages:
        if page.body_text and len(page.body_text) > 100:
            score = score_content(page.url, page.body_text)
            scores.append(score)
    return scores


def _find_duplicate_titles(pages: list[CrawledPage]) -> list[dict]:
    """Find pages with duplicate or very similar titles."""
    title_groups: dict[str, list[str]] = {}
    for page in pages:
        title = (page.title or "").strip().lower()
        if title:
            title_groups.setdefault(title, []).append(page.url)

    duplicates = []
    for title, urls in title_groups.items():
        if len(urls) > 1:
            duplicates.append({"title": title, "urls": urls})

    return duplicates


def _load_gsc_csv(path: str) -> list[dict]:
    """Load Google Search Console CSV export."""
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("gsc_csv_not_found", path=path)
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    logger.info("gsc_loaded", rows=len(rows))
    return rows


def _load_ahrefs_csv(path: str) -> list[dict]:
    """Load Ahrefs CSV export."""
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("ahrefs_csv_not_found", path=path)
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    logger.info("ahrefs_loaded", rows=len(rows))
    return rows


def _generate_recommendations(
    audit: SEOAuditResult,
    gsc_data: list[dict],
    ahrefs_data: list[dict],
) -> list[RecommendedFix]:
    """Generate prioritized recommendations from audit findings."""
    fixes = []

    # Cannibalization fixes
    for cluster in audit.cannibalization_clusters:
        fixes.append(RecommendedFix(
            priority=FixPriority.HIGH,
            action=f"Resolve cannibalization: consolidate or differentiate {len(cluster.pages)} competing pages",
            target_page=cluster.pages[0] if cluster.pages else "",
            expected_impact="Eliminates internal competition, concentrates ranking signals on one canonical page",
            effort="medium",
        ))

    # AI content fixes
    ai_flagged = [s for s in audit.ai_content_scores if s.verdict == "likely_ai"]
    for score in ai_flagged[:5]:  # Top 5
        fixes.append(RecommendedFix(
            priority=FixPriority.HIGH,
            action=f"Humanize content: add expert quotes, specific examples, original data",
            target_page=score.url,
            expected_impact="Reduces AI-gen deranking risk, improves E-E-A-T signals",
            effort="medium",
        ))

    # Schema fixes
    if audit.missing_schema_pages:
        fixes.append(RecommendedFix(
            priority=FixPriority.MEDIUM,
            action=f"Add FAQ/HowTo/Article schema markup to {len(audit.missing_schema_pages)} pages",
            target_page="multiple",
            expected_impact="Improves structured data signals for LLM citation and search rich results",
            effort="low",
        ))

    # Duplicate title fixes
    if audit.duplicate_titles:
        fixes.append(RecommendedFix(
            priority=FixPriority.MEDIUM,
            action=f"Fix {len(audit.duplicate_titles)} duplicate page titles — each page needs a unique, descriptive title",
            target_page="multiple",
            expected_impact="Helps search engines and LLMs distinguish between pages",
            effort="low",
        ))

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    fixes.sort(key=lambda f: priority_order.get(f.priority.value, 4))

    return fixes
