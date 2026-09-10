"""
Keyword Map & Gap Analysis for Citewatch.

Parses GSC/Ahrefs CSV exports, clusters keywords by embedding similarity,
and detects gaps where competitors rank but we don't.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import structlog

from citewatch.embeddings import get_embedding_manager
from citewatch.models import (
    KeywordCluster,
    KeywordEntry,
    KeywordGap,
    RecommendedFix,
    FixPriority,
)

logger = structlog.get_logger()


# ── CSV Parsers ───────────────────────────────────────────────────────────

def load_gsc_keywords(csv_path: str | Path) -> list[KeywordEntry]:
    """
    Load keywords from a Google Search Console CSV export.

    Expected columns (case-insensitive, flexible):
        query/keyword, page/url, clicks, impressions, ctr, position
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("gsc_csv_not_found", path=str(path))
        return []

    entries: list[KeywordEntry] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize header names
        fieldnames = {fn.strip().lower(): fn for fn in (reader.fieldnames or [])}

        query_col = fieldnames.get("query") or fieldnames.get("keyword") or fieldnames.get("top queries")
        url_col = fieldnames.get("page") or fieldnames.get("url") or fieldnames.get("landing page")
        clicks_col = fieldnames.get("clicks")
        impressions_col = fieldnames.get("impressions")
        ctr_col = fieldnames.get("ctr")
        position_col = fieldnames.get("position") or fieldnames.get("average position")

        if not query_col:
            logger.error("gsc_csv_no_query_column", columns=list(fieldnames.keys()))
            return []

        for row in reader:
            keyword = row.get(query_col, "").strip()
            if not keyword:
                continue
            entries.append(KeywordEntry(
                keyword=keyword,
                url=row.get(url_col, "").strip() if url_col else "",
                clicks=_safe_int(row.get(clicks_col, "0") if clicks_col else "0"),
                impressions=_safe_int(row.get(impressions_col, "0") if impressions_col else "0"),
                ctr=_safe_float(row.get(ctr_col, "0") if ctr_col else "0"),
                position=_safe_float(row.get(position_col, "0") if position_col else "0"),
                source="gsc",
            ))

    logger.info("gsc_loaded", keywords=len(entries))
    return entries


def load_ahrefs_keywords(csv_path: str | Path) -> list[KeywordEntry]:
    """
    Load keywords from an Ahrefs CSV export.

    Expected columns (case-insensitive, flexible):
        keyword, volume, difficulty/kd, url/current url, position/current position
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("ahrefs_csv_not_found", path=str(path))
        return []

    entries: list[KeywordEntry] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = {fn.strip().lower(): fn for fn in (reader.fieldnames or [])}

        keyword_col = fieldnames.get("keyword") or fieldnames.get("query")
        volume_col = fieldnames.get("volume") or fieldnames.get("search volume")
        difficulty_col = fieldnames.get("kd") or fieldnames.get("difficulty") or fieldnames.get("keyword difficulty")
        url_col = fieldnames.get("url") or fieldnames.get("current url") or fieldnames.get("page")
        position_col = fieldnames.get("position") or fieldnames.get("current position")

        if not keyword_col:
            logger.error("ahrefs_csv_no_keyword_column", columns=list(fieldnames.keys()))
            return []

        for row in reader:
            keyword = row.get(keyword_col, "").strip()
            if not keyword:
                continue
            entries.append(KeywordEntry(
                keyword=keyword,
                url=row.get(url_col, "").strip() if url_col else "",
                position=_safe_float(row.get(position_col, "0") if position_col else "0"),
                search_volume=_safe_int(row.get(volume_col, "0") if volume_col else "0"),
                difficulty=_safe_float(row.get(difficulty_col, "0") if difficulty_col else "0"),
                source="ahrefs",
            ))

    logger.info("ahrefs_loaded", keywords=len(entries))
    return entries


# ── Keyword Clustering ────────────────────────────────────────────────────

def cluster_keywords(
    keywords: list[KeywordEntry],
    similarity_threshold: float = 0.80,
) -> list[KeywordCluster]:
    """
    Group keywords into semantic clusters using embedding similarity.

    Keywords with cosine similarity > threshold are grouped together.
    The highest-volume keyword becomes the primary keyword.
    """
    if not keywords:
        return []

    emb = get_embedding_manager()
    texts = [kw.keyword for kw in keywords]
    embeddings = emb.embed_texts(texts)

    logger.info("clustering_keywords", total=len(keywords), threshold=similarity_threshold)

    visited: set[int] = set()
    clusters: list[KeywordCluster] = []

    for i in range(len(keywords)):
        if i in visited:
            continue

        cluster_indices = [i]
        visited.add(i)

        for j in range(i + 1, len(keywords)):
            if j in visited:
                continue
            sim = emb.cosine_similarity(embeddings[i], embeddings[j])
            if sim > similarity_threshold:
                cluster_indices.append(j)
                visited.add(j)

        # Build cluster
        cluster_kws = [keywords[idx] for idx in cluster_indices]
        cluster_texts = [kw.keyword for kw in cluster_kws]

        # Primary keyword = highest volume, falling back to most clicks
        primary = max(cluster_kws, key=lambda k: (k.search_volume, k.clicks, k.impressions))

        # Collect unique URLs
        our_pages = list({kw.url for kw in cluster_kws if kw.url and kw.source != "competitor"})
        competitor_pages = list({kw.url for kw in cluster_kws if kw.source == "competitor"})

        cluster_id = hashlib.sha256(primary.keyword.encode()).hexdigest()[:12]

        avg_pos = 0.0
        ranked = [kw.position for kw in cluster_kws if kw.position > 0 and kw.source != "competitor"]
        if ranked:
            avg_pos = sum(ranked) / len(ranked)

        total_vol = sum(kw.search_volume for kw in cluster_kws)

        clusters.append(KeywordCluster(
            cluster_id=cluster_id,
            primary_keyword=primary.keyword,
            keywords=cluster_texts,
            our_pages=our_pages,
            competitor_pages=competitor_pages,
            avg_position=avg_pos,
            total_volume=total_vol,
            is_cannibalized=len(our_pages) > 1,
        ))

    # Sort by total volume descending
    clusters.sort(key=lambda c: c.total_volume, reverse=True)

    logger.info("keyword_clustering_complete", clusters=len(clusters))
    return clusters


# ── Gap Detection ─────────────────────────────────────────────────────────

def detect_keyword_gaps(
    our_keywords: list[KeywordEntry],
    competitor_keywords: list[KeywordEntry],
    our_domain: str = "",
    similarity_threshold: float = 0.75,
) -> list[KeywordGap]:
    """
    Find keywords where competitors rank but we don't (or rank poorly).

    Uses embedding similarity to match competitor keywords to our closest pages.
    """
    if not competitor_keywords:
        return []

    emb = get_embedding_manager()

    # Build lookup of our keyword → position
    our_kw_map: dict[str, KeywordEntry] = {}
    for kw in our_keywords:
        if kw.keyword not in our_kw_map or kw.position < our_kw_map[kw.keyword].position:
            our_kw_map[kw.keyword.lower()] = kw

    # Embed our page titles/URLs for closest-page matching
    our_page_texts = list({kw.url for kw in our_keywords if kw.url})
    our_page_embeddings = emb.embed_texts(our_page_texts) if our_page_texts else []

    gaps: list[KeywordGap] = []

    for comp_kw in competitor_keywords:
        kw_lower = comp_kw.keyword.lower()

        # Check if we have this keyword
        our_entry = our_kw_map.get(kw_lower)

        # We have a gap if: we don't rank at all, or we rank much worse
        if our_entry and our_entry.position > 0 and our_entry.position <= 10:
            continue  # We already rank well for this

        # Find our closest page by embedding similarity
        closest_page = ""
        if our_page_texts:
            comp_emb = emb.embed_single(comp_kw.keyword)
            best_sim = 0.0
            for idx, page_emb in enumerate(our_page_embeddings):
                sim = emb.cosine_similarity(comp_emb, page_emb)
                if sim > best_sim:
                    best_sim = sim
                    closest_page = our_page_texts[idx]

        # Compute opportunity score
        volume_score = min(comp_kw.search_volume / 1000, 1.0) if comp_kw.search_volume > 0 else 0.3
        position_score = max(0, 1 - comp_kw.position / 50)
        difficulty_score = max(0, 1 - comp_kw.difficulty / 100) if comp_kw.difficulty > 0 else 0.5
        opportunity = (volume_score * 0.4 + position_score * 0.3 + difficulty_score * 0.3)

        # Classify difficulty
        diff_val = comp_kw.difficulty
        if diff_val > 70:
            diff_label = "high"
        elif diff_val > 40:
            diff_label = "medium"
        else:
            diff_label = "low"

        comp_domain = urlparse(comp_kw.url).netloc if comp_kw.url else ""

        gaps.append(KeywordGap(
            keyword=comp_kw.keyword,
            competitor_url=comp_kw.url,
            competitor_domain=comp_domain,
            competitor_position=comp_kw.position,
            our_closest_page=closest_page,
            our_position=our_entry.position if our_entry else 0.0,
            difficulty=diff_label,
            search_volume=comp_kw.search_volume,
            opportunity_score=round(opportunity, 3),
            recommendation=_gap_recommendation(comp_kw, our_entry, closest_page),
        ))

    # Sort by opportunity score descending
    gaps.sort(key=lambda g: g.opportunity_score, reverse=True)

    logger.info("keyword_gaps_detected", total=len(gaps))
    return gaps


def _gap_recommendation(comp_kw: KeywordEntry, our_kw: KeywordEntry | None, closest_page: str) -> str:
    """Generate a specific recommendation for a keyword gap."""
    if our_kw and our_kw.position > 0:
        return (
            f"Currently ranking at position {our_kw.position:.0f} for '{comp_kw.keyword}'. "
            f"Competitor ranks at {comp_kw.position:.0f}. "
            f"Improve content depth and authority signals on {our_kw.url or closest_page}."
        )
    elif closest_page:
        return (
            f"Not ranking for '{comp_kw.keyword}' — closest existing page is {closest_page}. "
            f"Consider adding a dedicated section or creating new content targeting this keyword."
        )
    else:
        return (
            f"No coverage for '{comp_kw.keyword}'. "
            f"Create new content targeting this keyword cluster."
        )


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_int(val: str) -> int:
    try:
        return int(float(val.replace(",", "").replace("%", "").strip()))
    except (ValueError, TypeError):
        return 0

def _safe_float(val: str) -> float:
    try:
        return float(val.replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return 0.0
