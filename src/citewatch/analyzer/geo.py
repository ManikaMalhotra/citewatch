"""
GEO Analyzer — RAG-augmented citation analysis engine.

Takes captured LLM responses, retrieves relevant knowledge-base chunks,
and uses an LLM judge to analyze citation patterns, competitor signals,
and generate actionable recommendations.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from citewatch.cache import KnowledgeBase
from citewatch.config import settings
from citewatch.models import (
    CitationStatus,
    CompetitorSignal,
    FixPriority,
    GEOAnalysis,
    LLMResponse,
    RecommendedFix,
)

logger = structlog.get_logger()


def _judge_system_prompt(cfg) -> str:
    """Build a brand-agnostic GEO judge prompt from config.yaml."""
    brand = cfg.target.name
    description = cfg.target.description or brand
    domain = cfg.target.domain
    context = cfg.target.domain_context or description
    competitors = ", ".join(c.name for c in cfg.competitors) or "(none configured)"
    return f"""You are a senior GEO (Generative Engine Optimization) auditor for {brand} ({domain}).

Product: {description}
Domain context: {context}

Your job: analyze LLM responses to determine why {brand} was cited or ignored, what authority signals competitors have that {brand} lacks, and what content changes would improve {brand}'s citation rate.

## Context
- Target brand: {brand} ({domain})
- Key competitors: {competitors}

## Your Task

Given:
1. The original user query
2. One or more LLM responses (with extracted brand mentions, citations, CoT)
3. Relevant chunks from the {brand} knowledge base

Analyze and output a JSON object with this exact schema:
{{
  "citation_status": "present" | "absent" | "partial",
  "brand_mentioned": true/false,
  "brand_recommended": true/false,
  "brand_context": "how {brand} was described, if mentioned",
  "why_brand_missing": "concise analysis if {brand} was not cited",
  "competitor_signals": [
    {{"brand": "...", "signal_type": "...", "detail": "...", "strength": "strong|moderate|weak"}}
  ],
  "authority_signals_missing": ["list of signals {brand} lacks"],
  "recommended_fixes": [
    {{"priority": "critical|high|medium|low", "action": "...", "target_page": "...", "expected_impact": "...", "effort": "low|medium|high"}}
  ],
  "confidence": 0.0-1.0,
  "raw_judge_reasoning": "brief reasoning summary in 2-4 sentences"
}}

Constraints:
- Return at most 6 competitor_signals
- Return at most 8 authority_signals_missing items
- Return at most 5 recommended_fixes
- Keep each competitor_signals.detail under 220 characters
- Keep each fix.action under 260 characters
- Keep why_brand_missing under 900 characters
- raw_judge_reasoning must be a short summary, not chain-of-thought

Be brutally honest. If {brand} deserves to be ignored because its content is thin, say so. Focus on actionable, specific fixes — not generic SEO advice.

Output ONLY valid JSON. No markdown fencing, no explanation outside the JSON."""


async def analyze_query(
    query: str,
    responses: list[LLMResponse],
    kb: KnowledgeBase,
    compare_brands: list[str] | None = None,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
) -> GEOAnalysis:
    """
    Run GEO analysis on captured LLM responses.

    1. Retrieve relevant KB chunks for context
    2. Build a comprehensive prompt with all response data
    3. Send to Claude Sonnet as GEO judge
    4. Parse structured output into GEOAnalysis
    """
    cfg = settings()
    competitors = compare_brands or cfg.get_competitor_names()

    # Step 1: Retrieve relevant KB chunks
    kb_results = kb.query(query, n_results=8)
    kb_chunks = []
    if kb_results and kb_results.get("documents"):
        kb_chunks = kb_results["documents"][0] if kb_results["documents"] else []
    kb_metadatas = []
    if kb_results and kb_results.get("metadatas"):
        kb_metadatas = kb_results["metadatas"][0] if kb_results["metadatas"] else []

    logger.info("geo_analysis_start", query=query[:60], responses=len(responses), kb_chunks=len(kb_chunks))

    # Step 2: Build the analysis prompt
    prompt = _build_analysis_prompt(
        query, responses, kb_chunks, kb_metadatas, competitors, brand_name=cfg.target.name,
    )

    # Step 3: Send to GEO judge LLM (Claude preferred, Gemini fallback)
    llm = _get_judge_llm(
        cfg,
        preferred_provider=preferred_provider,
        preferred_model_id=preferred_model_id,
    )

    if llm is None:
        logger.error("no_judge_model", msg="No API key set — cannot run GEO analysis")
        return _empty_analysis(query, responses)

    judge_response = await llm.ainvoke([
        SystemMessage(content=_judge_system_prompt(cfg)),
        HumanMessage(content=prompt),
    ])

    raw_judge = judge_response.content

    # Step 4: Parse into GEOAnalysis
    analysis = _parse_judge_response(query, responses, raw_judge)

    logger.info(
        "geo_analysis_complete",
        query=query[:60],
        citation_status=analysis.citation_status.value,
        fixes=len(analysis.recommended_fixes),
        confidence=analysis.confidence,
    )

    return analysis


def _build_analysis_prompt(
    query: str,
    responses: list[LLMResponse],
    kb_chunks: list[str],
    kb_metadatas: list[dict],
    competitors: list[str],
    brand_name: str = "target brand",
) -> str:
    """Build the comprehensive analysis prompt for the GEO judge."""
    parts = [f"## Query\n{query}\n"]

    # Add LLM responses
    parts.append("## LLM Responses\n")
    for i, resp in enumerate(responses):
        parts.append(f"### Response {i+1} (Model: {resp.model}, Provider: {resp.provider})")
        parts.append(f"**Raw response:**\n{resp.raw_text[:1200]}\n")

        if resp.mentioned_brands:
            brands_str = ", ".join(
                f"{b.brand} ({b.sentiment}, recommended={b.is_recommended})"
                for b in resp.mentioned_brands
            )
            parts.append(f"**Detected brands:** {brands_str}")

        if resp.citations:
            citations_str = "\n".join(f"- {c.title or c.url}" for c in resp.citations)
            parts.append(f"**Citations:**\n{citations_str}")

        parts.append("")

    # Add KB context
    if kb_chunks:
        parts.append(f"## {brand_name} Knowledge Base (Relevant Chunks)\n")
        for i, (chunk, meta) in enumerate(zip(kb_chunks, kb_metadatas)):
            url = meta.get("url", "unknown") if meta else "unknown"
            heading = meta.get("heading", "") if meta else ""
            parts.append(f"### KB Chunk {i+1} (Source: {url})")
            if heading:
                parts.append(f"**Section:** {heading}")
            parts.append(f"{chunk[:700]}\n")
    else:
        parts.append(f"## {brand_name} Knowledge Base\n*No KB chunks available. The knowledge base may not be ingested yet.*\n")

    # Add competitor context
    parts.append(f"## Competitors to Track\n{', '.join(competitors)}\n")

    return "\n".join(parts)


def _parse_judge_response(
    query: str,
    responses: list[LLMResponse],
    raw_judge: str,
) -> GEOAnalysis:
    """Parse the judge LLM's JSON response into a GEOAnalysis object."""
    # Try to extract JSON from the response
    json_str = raw_judge.strip()

    # Handle markdown-fenced JSON
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("judge_json_parse_failed", raw=raw_judge[:200])
        return _empty_analysis(query, responses, raw_judge)

    # Build GEOAnalysis from parsed data
    competitor_signals = []
    for cs in data.get("competitor_signals", []):
        competitor_signals.append(CompetitorSignal(
            brand=cs.get("brand", ""),
            signal_type=cs.get("signal_type", ""),
            detail=cs.get("detail", ""),
            strength=cs.get("strength", "moderate"),
        ))

    recommended_fixes = []
    for fix in data.get("recommended_fixes", []):
        try:
            recommended_fixes.append(RecommendedFix(
                priority=FixPriority(fix.get("priority", "medium")),
                action=fix.get("action", ""),
                target_page=fix.get("target_page", ""),
                expected_impact=fix.get("expected_impact", ""),
                effort=fix.get("effort", "medium"),
            ))
        except (ValueError, KeyError):
            continue

    return GEOAnalysis(
        query=query,
        models_analyzed=[r.model for r in responses],
        citation_status=CitationStatus(data.get("citation_status", "absent")),
        brand_mentioned=bool(data.get("brand_mentioned", False)),
        brand_recommended=bool(data.get("brand_recommended", False)),
        brand_context=data.get("brand_context") or "",
        why_brand_missing=data.get("why_brand_missing") or "",
        competitor_signals=competitor_signals,
        authority_signals_missing=data.get("authority_signals_missing", []),
        recommended_fixes=recommended_fixes,
        confidence=float(data.get("confidence", 0.5)),
        raw_judge_reasoning=data.get("raw_judge_reasoning") or "",
    )


def _get_judge_llm(
    cfg,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
):
    """Get the best available LLM for the GEO judge."""
    if preferred_model_id:
        preferred_model = next(
            (m for m in cfg.models.get_all() if m.model_id == preferred_model_id),
            None,
        )
        if preferred_model:
            if preferred_model.provider == "anthropic" and cfg.anthropic_api_key:
                from langchain_anthropic import ChatAnthropic
                logger.info("judge_llm", provider="anthropic", model=preferred_model.model_id)
                return ChatAnthropic(
                    model=preferred_model.model_id,
                    api_key=cfg.anthropic_api_key,
                    temperature=0.0,
                    max_tokens=4096,
                )
            if preferred_model.provider == "google" and cfg.google_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info("judge_llm", provider="google", model=preferred_model.model_id)
                return ChatGoogleGenerativeAI(
                    model=preferred_model.model_id,
                    google_api_key=cfg.google_api_key,
                    temperature=0.0,
                    max_output_tokens=4096,
                )

    provider_order = ["anthropic", "google"]
    if preferred_provider in provider_order:
        provider_order.remove(preferred_provider)
        provider_order.insert(0, preferred_provider)

    for provider in provider_order:
        if provider == "anthropic" and cfg.anthropic_api_key and cfg.models.claude:
            model = None
            for m in cfg.models.claude:
                if m.tier == "mid":
                    model = m
                    break
            if not model:
                model = cfg.models.claude[0]
            from langchain_anthropic import ChatAnthropic
            logger.info("judge_llm", provider="anthropic", model=model.model_id)
            return ChatAnthropic(model=model.model_id, api_key=cfg.anthropic_api_key, temperature=0.0, max_tokens=4096)

        if provider == "google" and cfg.google_api_key and cfg.models.gemini:
            model = None
            for m in cfg.models.gemini:
                if m.tier == "high":
                    model = m
                    break
            if not model:
                model = cfg.models.gemini[0]
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("judge_llm", provider="google", model=model.model_id)
            return ChatGoogleGenerativeAI(model=model.model_id, google_api_key=cfg.google_api_key, temperature=0.0, max_output_tokens=4096)

    return None


def _empty_analysis(
    query: str,
    responses: list[LLMResponse],
    raw_judge: str = "",
) -> GEOAnalysis:
    """Return an empty analysis when parsing fails."""
    return GEOAnalysis(
        query=query,
        models_analyzed=[r.model for r in responses],
        citation_status=CitationStatus.ABSENT,
        why_brand_missing="Analysis could not be completed — judge response parsing failed",
        confidence=0.0,
        raw_judge_reasoning=raw_judge,
    )
