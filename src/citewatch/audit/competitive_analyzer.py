"""
LLM-powered competitive article analysis for Citewatch.

Uses the same provider-agnostic LLM factory as the GEO judge to generate
detailed head-to-head article comparisons and actionable recommendations.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from citewatch.config import settings
from citewatch.models import (
    ArticleComparison,
    CompetitorPage,
    CrawledPage,
    KeywordCluster,
    RecommendedFix,
    FixPriority,
)

logger = structlog.get_logger()


COMPETITIVE_SYSTEM_PROMPT = """You are a senior SEO & content strategist analyzing why competitor articles outrank the target brand's content.

## Your Role
- Compare our content against competitor content for the same keyword/topic
- Identify specific, actionable reasons why the competitor ranks better
- Provide concrete improvement recommendations with examples

## Analysis Framework
For each comparison, evaluate:
1. **Content depth**: Word count, topic coverage, subtopics addressed
2. **Structure**: Headings, internal links, table of contents, FAQ sections
3. **Authority signals**: Data, benchmarks, case studies, expert quotes, citations
4. **Technical accuracy**: Code examples, architecture diagrams, implementation guides
5. **Schema markup**: Structured data presence (FAQ, HowTo, Article, etc.)
6. **User intent match**: Does the content fully answer the user's likely questions?

## Output Format
Respond with a JSON object:
```json
{
    "strengths_theirs": ["specific strength 1", "specific strength 2", ...],
    "gaps_ours": ["specific gap 1", "specific gap 2", ...],
    "recommended_actions": [
        "Action 1: specific, actionable recommendation",
        "Action 2: another specific recommendation",
        ...
    ],
    "summary": "2-3 sentence executive summary of why they rank better and what we should do"
}
```

Be SPECIFIC. Don't say "improve content quality." Instead say "Add a comparison table covering 10+ tools with columns for pricing, OTel support, and deployment model — competitor has this at line 45."
"""


async def analyze_article_pair(
    our_page: CrawledPage | None,
    competitor_page: CompetitorPage,
    keyword: str,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
) -> ArticleComparison:
    """
    Generate a detailed head-to-head comparison of our page vs a competitor's.

    Uses the GEO judge LLM to analyze content differences and generate recommendations.
    """
    cfg = settings()

    # Get the LLM (reuse the same factory as the GEO judge)
    from citewatch.analyzer.geo import _get_judge_llm
    llm = _get_judge_llm(
        cfg,
        preferred_provider=preferred_provider,
        preferred_model_id=preferred_model_id,
    )

    if llm is None:
        logger.error("no_llm_for_competitive", msg="No API key set — cannot run competitive analysis")
        return ArticleComparison(
            keyword_cluster=keyword,
            our_url=our_page.url if our_page else "",
            competitor_url=competitor_page.url,
            competitor_domain=competitor_page.domain,
            gaps_ours=["Cannot analyze: no LLM API key configured"],
        )

    # Build the comparison prompt
    prompt = _build_comparison_prompt(our_page, competitor_page, keyword)

    try:
        response = await llm.ainvoke([
            SystemMessage(content=COMPETITIVE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw_text = response.content
        parsed = _parse_comparison_response(raw_text)

        return ArticleComparison(
            keyword_cluster=keyword,
            our_url=our_page.url if our_page else "",
            our_title=our_page.title if our_page else "",
            competitor_url=competitor_page.url,
            competitor_domain=competitor_page.domain,
            competitor_title=competitor_page.title,
            content_similarity=0.0,  # Set by caller
            word_count_ours=our_page.word_count if our_page else 0,
            word_count_theirs=competitor_page.word_count,
            strengths_theirs=parsed.get("strengths_theirs", []),
            gaps_ours=parsed.get("gaps_ours", []),
            recommended_actions=parsed.get("recommended_actions", []),
            raw_analysis=parsed.get("summary", raw_text[:500]),
        )

    except Exception as e:
        logger.error("competitive_analysis_failed", keyword=keyword, error=str(e)[:200])
        return ArticleComparison(
            keyword_cluster=keyword,
            our_url=our_page.url if our_page else "",
            competitor_url=competitor_page.url,
            competitor_domain=competitor_page.domain,
            gaps_ours=[f"Analysis failed: {str(e)[:100]}"],
        )


async def analyze_competitive_batch(
    matches: list[tuple[CrawledPage, CompetitorPage, float, str]],
    top_n: int = 20,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
) -> list[ArticleComparison]:
    """
    Analyze the top-N article matches in sequence (to respect rate limits).

    Args:
        matches: List of (our_page, competitor_page, similarity, cluster_keyword) tuples.
        top_n: Number of comparisons to generate.
    """
    comparisons: list[ArticleComparison] = []

    for our_page, comp_page, similarity, cluster_kw in matches[:top_n]:
        logger.info(
            "analyzing_pair",
            ours=our_page.url[:60],
            theirs=comp_page.url[:60],
            sim=f"{similarity:.3f}",
        )

        comparison = await analyze_article_pair(
            our_page,
            comp_page,
            keyword=cluster_kw or our_page.title,
            preferred_provider=preferred_provider,
            preferred_model_id=preferred_model_id,
        )
        comparison.content_similarity = float(similarity)
        comparisons.append(comparison)

    logger.info("competitive_batch_complete", comparisons=len(comparisons))
    return comparisons


def generate_competitive_recommendations(
    comparisons: list[ArticleComparison],
    keyword_gaps: list,
    cannibalization: list,
) -> list[RecommendedFix]:
    """Generate prioritized recommendations from all competitive analysis data."""
    fixes: list[RecommendedFix] = []

    # From article comparisons
    for comp in comparisons:
        if comp.recommended_actions:
            # Top action becomes a fix
            action = comp.recommended_actions[0]
            fixes.append(RecommendedFix(
                priority=FixPriority.HIGH if comp.word_count_theirs > comp.word_count_ours * 1.5 else FixPriority.MEDIUM,
                action=f"[{comp.keyword_cluster}] {action}",
                target_page=comp.our_url or "new page needed",
                expected_impact=(
                    f"Competitor ({comp.competitor_domain}) has {comp.word_count_theirs} words vs our "
                    f"{comp.word_count_ours}. Closing this gap should improve ranking."
                ),
                effort="high" if comp.word_count_theirs > 2000 else "medium",
            ))

    # From keyword gaps (top 10)
    from citewatch.models import KeywordGap
    for gap in keyword_gaps[:10]:
        if isinstance(gap, KeywordGap):
            fixes.append(RecommendedFix(
                priority=FixPriority.CRITICAL if gap.opportunity_score > 0.7 else FixPriority.HIGH,
                action=f"Create content targeting '{gap.keyword}' — competitor ranks at position {gap.competitor_position:.0f}",
                target_page=gap.our_closest_page or f"new page targeting '{gap.keyword}'",
                expected_impact=gap.recommendation,
                effort="high",
            ))

    # From cannibalization
    for cluster in cannibalization:
        if hasattr(cluster, 'pages') and len(cluster.pages) > 1:
            fixes.append(RecommendedFix(
                priority=FixPriority.HIGH,
                action=f"Resolve cannibalization: {len(cluster.pages)} pages compete for the same keywords",
                target_page=cluster.pages[0],
                expected_impact="Consolidating internal competition concentrates ranking signals on one canonical page",
                effort="medium",
            ))

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    fixes.sort(key=lambda f: priority_order.get(f.priority.value, 4))

    return fixes


def _build_comparison_prompt(
    our_page: CrawledPage | None,
    competitor_page: CompetitorPage,
    keyword: str,
) -> str:
    """Build the comparison prompt for the LLM."""
    parts = [f"## Target Keyword/Topic\n{keyword}\n"]

    # Our content
    if our_page and our_page.body_text:
        our_host = ""
        if our_page and our_page.url:
            our_host = urlparse(our_page.url).netloc or "our site"
        parts.append(f"## OUR CONTENT ({our_host or 'our site'})")
        parts.append(f"**URL:** {our_page.url}")
        parts.append(f"**Title:** {our_page.title}")
        parts.append(f"**Word count:** {our_page.word_count}")
        if our_page.headings:
            parts.append(f"**Headings:**")
            for h in our_page.headings[:20]:
                parts.append(f"  - {h}")
        parts.append(f"\n**Content (first 3000 chars):**\n{our_page.body_text[:3000]}\n")
    else:
        parts.append(f"## OUR CONTENT\n*We have NO page targeting this keyword.*\n")
        parts.append("This means we need to create entirely new content.\n")

    # Competitor content
    parts.append(f"## COMPETITOR CONTENT ({competitor_page.domain})")
    parts.append(f"**URL:** {competitor_page.url}")
    parts.append(f"**Title:** {competitor_page.title}")
    parts.append(f"**Word count:** {competitor_page.word_count}")
    parts.append(f"**Has schema markup:** {'Yes' if competitor_page.has_schema_markup else 'No'}")
    if competitor_page.headings:
        parts.append(f"**Headings:**")
        for h in competitor_page.headings[:20]:
            parts.append(f"  - {h}")
    parts.append(f"\n**Content (first 3000 chars):**\n{competitor_page.body_text[:3000]}\n")

    parts.append("## Instructions")
    parts.append("Analyze both articles. Explain specifically why the competitor content is stronger.")
    parts.append("Focus on ACTIONABLE differences — things we can change to outrank them.")
    parts.append("Return your analysis as the JSON format specified in the system prompt.")

    return "\n".join(parts)


def _parse_comparison_response(raw: str) -> dict:
    """Parse the LLM's JSON response."""
    # Try to extract JSON from markdown fences
    text = raw.strip()

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("competitive_parse_failed", raw_length=len(raw))
    return {
        "strengths_theirs": ["Could not parse LLM response"],
        "gaps_ours": ["Manual review needed"],
        "recommended_actions": ["Re-run analysis or review raw output"],
        "summary": raw[:300],
    }
