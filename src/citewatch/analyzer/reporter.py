"""
Report generator for Citewatch.

Renders GEOAnalysis results into Rich Markdown reports
and structured JSON files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from citewatch.models import GEOAnalysis, LLMResponse, SEOAuditResult, CompetitiveAuditResult
from citewatch.config import settings

logger = structlog.get_logger()


def generate_geo_report(
    query: str,
    analysis: GEOAnalysis,
    responses: list[LLMResponse],
    output_dir: str = "./reports",
    *,
    report_status: str = "complete",
    run_notes: list[str] | None = None,
    capture_failures: list[dict] | None = None,
    analysis_error: str = "",
    intended_models: list[str] | None = None,
    expansions: list[dict] | None = None,
) -> dict[str, str]:
    """
    Generate a complete GEO analysis report.

    Returns:
        Dict with keys 'markdown' and 'json' pointing to file paths.
    """
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = Path(output_dir) / date_str
    report_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize query for filename
    safe_query = "".join(c if c.isalnum() or c in " -_" else "_" for c in query[:50]).strip().replace(" ", "_")
    base_name = f"geo_{safe_query}"

    # Generate Markdown report
    md_content = _render_markdown(
        query,
        analysis,
        responses,
        report_status=report_status,
        run_notes=run_notes or [],
        capture_failures=capture_failures or [],
        analysis_error=analysis_error,
        intended_models=intended_models or [],
    )
    md_path = report_dir / f"{base_name}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Generate JSON report
    json_data = {
        "query": query,
        "generated_at": datetime.utcnow().isoformat(),
        "report_status": report_status,
        "run_notes": run_notes or [],
        "capture_failures": capture_failures or [],
        "analysis_error": analysis_error,
        "intended_models": intended_models or [],
        "expansions": expansions or [],
        "analysis": analysis.model_dump(mode="json"),
        "responses": [r.model_dump(mode="json", exclude={"embedding"}) for r in responses],
    }
    json_path = report_dir / f"{base_name}.json"
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")

    logger.info("report_generated", md=str(md_path), json=str(json_path))
    return {"markdown": str(md_path), "json": str(json_path), "report": json_data}


def _render_markdown(
    query: str,
    analysis: GEOAnalysis,
    responses: list[LLMResponse],
    *,
    report_status: str = "complete",
    run_notes: list[str],
    capture_failures: list[dict],
    analysis_error: str,
    intended_models: list[str],
) -> str:
    """Render the GEO analysis as a Markdown report."""
    models_display = ", ".join(dict.fromkeys(analysis.models_analyzed or intended_models)) or "(none)"
    lines = [
        f"# GEO Analysis Report",
        f"",
        f"**Query:** {query}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Run Status:** {report_status.upper()}",
        f"**Models Analyzed:** {models_display}",
        f"**Responses Captured:** {len(responses)}",
        f"**Confidence:** {analysis.confidence:.0%}",
        f"",
        f"---",
        f"",
    ]

    if run_notes:
        lines.extend(["## Run Notes", ""])
        for note in run_notes:
            lines.append(f"- {note}")
        lines.append("")

    if capture_failures:
        lines.extend([
            "## Capture Issues",
            "",
            "| Phase | Model | Query | Type | Message |",
            "|-------|-------|-------|------|---------|",
        ])
        for failure in capture_failures:
            lines.append(
                f"| {failure.get('phase', '')} | {failure.get('model', '')} | "
                f"{failure.get('query', '')[:50]} | {failure.get('error_type', '')} | "
                f"{failure.get('message', '')[:120]} |"
            )
        lines.append("")

    if analysis_error:
        lines.extend([
            "## Analysis Status",
            "",
            f"Analysis could not be completed: {analysis_error}",
            "",
        ])

    lines.extend([
        f"## Citation Status: {_status_emoji(analysis.citation_status.value)} {analysis.citation_status.value.upper()}",
        f"",
    ])

    # Target brand status
    brand = settings().target.name
    if analysis.brand_mentioned:
        lines.append(f"✅ **{brand} was mentioned** in LLM responses")
        if analysis.brand_recommended:
            lines.append(f"⭐ **{brand} was recommended** as a solution")
        if analysis.brand_context:
            lines.append(f"\n> {analysis.brand_context}")
    else:
        lines.append(f"❌ **{brand} was NOT mentioned** in any LLM response")

    lines.append("")

    # Why missing
    if analysis.why_brand_missing:
        lines.extend([
            f"## Why {brand} Was Not Cited",
            f"",
            analysis.why_brand_missing,
            f"",
        ])

    # Competitor signals
    if analysis.competitor_signals:
        lines.extend([
            f"## Competitor Analysis",
            f"",
            f"| Brand | Signal Type | Detail | Strength |",
            f"|-------|------------|--------|----------|",
        ])
        for cs in analysis.competitor_signals:
            lines.append(f"| {cs.brand} | {cs.signal_type} | {cs.detail} | {_strength_emoji(cs.strength)} {cs.strength} |")
        lines.append("")

    # Missing authority signals
    if analysis.authority_signals_missing:
        lines.extend([
            f"## Missing Authority Signals",
            f"",
        ])
        for signal in analysis.authority_signals_missing:
            lines.append(f"- ⚠️ {signal}")
        lines.append("")

    # Recommended fixes
    if analysis.recommended_fixes:
        lines.extend([
            f"## Recommended Fixes",
            f"",
        ])
        for i, fix in enumerate(analysis.recommended_fixes, 1):
            emoji = _priority_emoji(fix.priority.value)
            lines.extend([
                f"### {emoji} Fix {i}: {fix.action}",
                f"",
                f"- **Priority:** {fix.priority.value}",
                f"- **Effort:** {fix.effort}",
                f"- **Target:** {fix.target_page or 'General'}",
                f"- **Expected Impact:** {fix.expected_impact}",
                f"",
            ])

    # Brand mentions across responses
    lines.extend([
        f"## Brand Mentions Across All Responses",
        f"",
    ])
    all_brands: dict[str, list] = {}
    for resp in responses:
        for bm in resp.mentioned_brands:
            all_brands.setdefault(bm.brand, []).append({
                "model": resp.model,
                "sentiment": bm.sentiment,
                "recommended": bm.is_recommended,
            })

    if all_brands:
        lines.extend([
            f"| Brand | Models | Sentiment | Recommended |",
            f"|-------|--------|-----------|-------------|",
        ])
        for brand, mentions in sorted(all_brands.items()):
            models = ", ".join(set(m["model"].split("/")[-1] for m in mentions))
            sentiments = ", ".join(set(m["sentiment"] for m in mentions))
            recommended = "✅" if any(m["recommended"] for m in mentions) else "—"
            lines.append(f"| {brand} | {models} | {sentiments} | {recommended} |")
    else:
        lines.append("*No brand mentions detected.*")

    lines.extend(["", "---", f"*Report generated by Citewatch v0.1.0*"])

    return "\n".join(lines)


def generate_audit_report(
    audit: SEOAuditResult,
    output_dir: str = "./reports",
) -> dict[str, str]:
    """Generate an SEO audit report."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = Path(output_dir) / date_str
    report_dir.mkdir(parents=True, exist_ok=True)

    md_content = _render_audit_markdown(audit)
    md_path = report_dir / "seo_audit.md"
    md_path.write_text(md_content, encoding="utf-8")

    json_path = report_dir / "seo_audit.json"
    json_path.write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("audit_report_generated", md=str(md_path))
    return {"markdown": str(md_path), "json": str(json_path)}


def _render_audit_markdown(audit: SEOAuditResult) -> str:
    lines = [
        "# SEO Audit Report",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Pages Analyzed:** {audit.total_pages_analyzed}",
        "",
        "---",
        "",
    ]

    if audit.cannibalization_clusters:
        lines.extend(["## Cannibalization Clusters", ""])
        for i, cluster in enumerate(audit.cannibalization_clusters, 1):
            lines.append(f"### Cluster {i} (Similarity: {cluster.max_similarity:.2f})")
            lines.append(f"**Queries:** {', '.join(cluster.queries[:5])}")
            for page in cluster.pages:
                lines.append(f"- {page}")
            if cluster.recommendation:
                lines.append(f"\n> 💡 {cluster.recommendation}")
            lines.append("")

    if audit.ai_content_scores:
        flagged = [s for s in audit.ai_content_scores if s.verdict != "human"]
        if flagged:
            lines.extend(["## AI-Generated Content Flags", "", "| URL | Score | Verdict | Signals |", "|-----|-------|---------|---------|"])
            for score in flagged:
                signals = ", ".join(score.signals[:3])
                lines.append(f"| {score.url[:60]} | {score.overall_score:.2f} | {score.verdict} | {signals} |")
            lines.append("")

    if audit.missing_schema_pages:
        lines.extend(["## Pages Missing Schema Markup", ""])
        for url in audit.missing_schema_pages[:20]:
            lines.append(f"- {url}")
        lines.append("")

    if audit.top_recommendations:
        lines.extend(["## Top Recommendations", ""])
        for i, fix in enumerate(audit.top_recommendations, 1):
            lines.append(f"{i}. **[{fix.priority.value.upper()}]** {fix.action}")
        lines.append("")

    lines.extend(["---", "*Report generated by Citewatch v0.1.0*"])
    return "\n".join(lines)


def generate_competitive_report(
    result: CompetitiveAuditResult,
    output_dir: str = "./reports",
) -> dict[str, str]:
    """Generate a competitive SEO analysis report."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = Path(output_dir) / date_str
    report_dir.mkdir(parents=True, exist_ok=True)

    md_content = _render_competitive_markdown(result)
    md_path = report_dir / "competitive_analysis.md"
    md_path.write_text(md_content, encoding="utf-8")

    json_path = report_dir / "competitive_analysis.json"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("competitive_report_generated", md=str(md_path))
    return {"markdown": str(md_path), "json": str(json_path)}


def _render_competitive_markdown(result: CompetitiveAuditResult) -> str:
    """Render competitive audit as Markdown report."""
    lines = [
        "# Competitive SEO Analysis Report",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Domains Analyzed:** {', '.join(result.domains_analyzed) or '(none)'}",
        f"**Keywords Analyzed:** {result.total_keywords_analyzed}",
        f"**Keyword Gaps Found:** {len(result.keyword_gaps)}",
        f"**Article Comparisons:** {len(result.article_comparisons)}",
        "",
        "---",
        "",
    ]

    # ── Keyword Map ──────────────────────────────────────────────────
    if result.keyword_map:
        lines.extend([
            "## 📊 Keyword Map — Our Current Coverage",
            "",
            "| Keyword | URL | Position | Clicks | Impressions | Volume | Source |",
            "|---------|-----|----------|--------|-------------|--------|--------|",
        ])
        for kw in sorted(result.keyword_map, key=lambda k: k.clicks, reverse=True)[:50]:
            pos_str = f"{kw.position:.0f}" if kw.position > 0 else "—"
            lines.append(
                f"| {kw.keyword} | {_truncate_url(kw.url)} | {pos_str} | "
                f"{kw.clicks} | {kw.impressions} | {kw.search_volume} | {kw.source} |"
            )
        if len(result.keyword_map) > 50:
            lines.append(f"\n*Showing top 50 of {len(result.keyword_map)} keywords.*")
        lines.append("")

    # ── Keyword Clusters ─────────────────────────────────────────────
    if result.keyword_clusters:
        lines.extend([
            "## 🎯 Keyword Clusters",
            "",
        ])
        for cluster in result.keyword_clusters[:20]:
            cannib = " ⚠️ CANNIBALIZED" if cluster.is_cannibalized else ""
            lines.append(f"### {cluster.primary_keyword}{cannib}")
            lines.append(f"- **Keywords:** {', '.join(cluster.keywords[:8])}")
            if cluster.our_pages:
                lines.append(f"- **Our pages:** {', '.join(_truncate_url(u) for u in cluster.our_pages[:3])}")
            if cluster.competitor_pages:
                lines.append(f"- **Competitor pages:** {', '.join(_truncate_url(u) for u in cluster.competitor_pages[:3])}")
            lines.append(f"- **Avg position:** {cluster.avg_position:.0f}" if cluster.avg_position > 0 else "- **Avg position:** Not ranking")
            lines.append(f"- **Total volume:** {cluster.total_volume}")
            lines.append("")

    # ── Keyword Gaps ─────────────────────────────────────────────────
    if result.keyword_gaps:
        lines.extend([
            "## 🔍 Keyword Gaps — Competitors Rank, We Don't",
            "",
            "| Keyword | Competitor | Their Pos | Our Pos | Volume | Difficulty | Opportunity |",
            "|---------|-----------|-----------|---------|--------|------------|-------------|",
        ])
        for gap in result.keyword_gaps[:30]:
            our_pos = f"{gap.our_position:.0f}" if gap.our_position > 0 else "—"
            lines.append(
                f"| {gap.keyword} | {gap.competitor_domain} | "
                f"{gap.competitor_position:.0f} | {our_pos} | "
                f"{gap.search_volume} | {gap.difficulty} | "
                f"{gap.opportunity_score:.2f} |"
            )
        lines.append("")

        # Top gap recommendations
        lines.extend(["### Top Gap Recommendations", ""])
        for gap in result.keyword_gaps[:5]:
            lines.append(f"- **{gap.keyword}**: {gap.recommendation}")
        lines.append("")

    # ── Cannibalization ──────────────────────────────────────────────
    if result.cannibalization_clusters:
        lines.extend([
            "## ⚠️ Cannibalization Issues",
            "",
        ])
        for i, cluster in enumerate(result.cannibalization_clusters, 1):
            lines.append(f"### Cluster {i} (Similarity: {cluster.max_similarity:.2f})")
            lines.append(f"**Competing keywords:** {', '.join(cluster.queries[:5])}")
            for page in cluster.pages:
                lines.append(f"- {page}")
            if cluster.recommendation:
                lines.append(f"\n> 💡 {cluster.recommendation}")
            lines.append("")

    # ── Article Comparisons ──────────────────────────────────────────
    if result.article_comparisons:
        lines.extend([
            "## 📝 Article-Level Competitive Analysis",
            "",
        ])
        for comp in result.article_comparisons:
            lines.extend([
                f"### {comp.keyword_cluster}",
                "",
                f"**Our page:** {comp.our_url or '❌ No page exists'} ({comp.word_count_ours} words)",
                f"**Competitor:** {comp.competitor_url} ({comp.word_count_theirs} words) — {comp.competitor_domain}",
                f"**Content similarity:** {comp.content_similarity:.2f}",
                "",
            ])

            if comp.strengths_theirs:
                lines.append("**Why they rank better:**")
                for s in comp.strengths_theirs:
                    lines.append(f"- {s}")
                lines.append("")

            if comp.gaps_ours:
                lines.append("**What our content is missing:**")
                for g in comp.gaps_ours:
                    lines.append(f"- ❌ {g}")
                lines.append("")

            if comp.recommended_actions:
                lines.append("**What to do:**")
                for j, action in enumerate(comp.recommended_actions, 1):
                    lines.append(f"{j}. {action}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── Priority Action Items ────────────────────────────────────────
    if result.top_recommendations:
        lines.extend([
            "## 🚨 Priority Action Items",
            "",
            "| # | Priority | Action | Target | Impact | Effort |",
            "|---|----------|--------|--------|--------|--------|",
        ])
        for i, fix in enumerate(result.top_recommendations[:20], 1):
            emoji = _priority_emoji(fix.priority.value)
            lines.append(
                f"| {i} | {emoji} {fix.priority.value.upper()} | "
                f"{fix.action} | {_truncate_url(fix.target_page)} | "
                f"{fix.expected_impact[:80]} | {fix.effort} |"
            )
        lines.append("")

    lines.extend(["---", "*Report generated by Citewatch v0.1.0*"])
    return "\n".join(lines)


def _truncate_url(url: str, max_len: int = 50) -> str:
    """Shorten a URL for table display."""
    if not url or len(url) <= max_len:
        return url
    # Keep domain + last path segment
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    last_seg = path.rsplit("/", 1)[-1] if "/" in path else path
    return f"{parsed.netloc}/.../{last_seg}"


def _status_emoji(status: str) -> str:
    return {"present": "✅", "absent": "❌", "partial": "⚠️"}.get(status, "❓")

def _strength_emoji(strength: str) -> str:
    return {"strong": "🔴", "moderate": "🟡", "weak": "🟢"}.get(strength, "⚪")

def _priority_emoji(priority: str) -> str:
    return {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
