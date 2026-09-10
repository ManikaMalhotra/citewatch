"""
Query Expander for Citewatch.

Takes a seed query and generates semantically diverse variants
using Ollama embeddings + LLM expansion. Filters by cosine
diversity to avoid near-duplicate variants.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from citewatch.config import settings
from citewatch.embeddings import get_embedding_manager
from citewatch.models import QueryVariant

logger = structlog.get_logger()

EXPANDER_SYSTEM_PROMPT = """You are a search query expansion expert specializing in: {domain_context}

Given a seed query, generate exactly {num_variants} semantically diverse query variants. Each variant MUST target a different intent category:

Categories:
1. **beginner** — How a newcomer would phrase the question
2. **troubleshooting** — A developer debugging a specific problem
3. **comparison** — Comparing tools/approaches
4. **best_tool** — Looking for the best solution
5. **enterprise** — An engineering manager or architect evaluating at scale

Rules:
- Each variant must be a realistic search query a real person would type
- Variants should differ meaningfully in phrasing, not just word swaps
- Keep each variant under 20 words
- Domain context: {domain_context}

Output format (one per line, prefixed with category):
beginner: <query>
troubleshooting: <query>
comparison: <query>
best_tool: <query>
enterprise: <query>
"""


async def expand_query(
    seed_query: str,
    num_variants: int | None = None,
    diversity_threshold: float | None = None,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
) -> list[QueryVariant]:
    """
    Expand a seed query into diverse variants.

    1. Embed the seed query
    2. Use the preferred low-tier LLM to generate N variants across intent categories
    3. Embed all variants
    4. Filter by cosine diversity (reject variants too similar to each other)
    """
    cfg = settings()
    num = num_variants or cfg.expander.num_variants
    threshold = diversity_threshold or cfg.expander.diversity_threshold
    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)

    logger.info("expanding_query", seed=seed_query, num=num)

    # Step 1: Embed seed query
    seed_embedding = emb.embed_single(seed_query)

    # Step 2: Generate variants via the preferred available LLM
    llm = _get_expander_llm(
        cfg,
        preferred_provider=preferred_provider,
        preferred_model_id=preferred_model_id,
    )

    if llm is None:
        logger.warning("no_llm_for_expansion", msg="No API key set — using template fallback")
        return _fallback_expansion(seed_query, seed_embedding, emb)

    domain_context = cfg.target.domain_context or cfg.target.description or cfg.target.name
    system_msg = SystemMessage(content=EXPANDER_SYSTEM_PROMPT.format(
        num_variants=num,
        domain_context=domain_context,
    ))
    human_msg = HumanMessage(content=f"Seed query: {seed_query}")

    response = await llm.ainvoke([system_msg, human_msg])
    raw_text = response.content

    # Step 3: Parse variants
    variants = _parse_variants(raw_text, seed_query)
    if not variants:
        logger.warning("variant_parsing_failed", raw=raw_text[:200])
        return _fallback_expansion(seed_query, seed_embedding, emb)

    # Step 4: Embed all variants
    variant_texts = [v.variant_text for v in variants]
    variant_embeddings = emb.embed_texts(variant_texts)

    for v, vec in zip(variants, variant_embeddings):
        v.embedding = vec
        v.similarity_to_seed = emb.cosine_similarity(seed_embedding, vec)

    # Step 5: Diversity filtering — reject variants too similar to each other
    filtered = _diversity_filter(variants, emb, threshold)

    logger.info(
        "expansion_complete",
        seed=seed_query,
        generated=len(variants),
        after_filter=len(filtered),
    )
    return filtered


def _parse_variants(raw_text: str, seed_query: str) -> list[QueryVariant]:
    """Parse LLM output into QueryVariant objects."""
    variants = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Handle formats like "1. beginner: ..." or "beginner: ..." or "**beginner**: ..."
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        category = parts[0].strip().lower()
        # Clean category: remove numbering, asterisks, dashes
        for char in "0123456789.-*# ":
            category = category.strip(char)
        variant_text = parts[1].strip().strip('"').strip("'")
        if not variant_text or len(variant_text) < 5:
            continue
        if category not in ("beginner", "troubleshooting", "comparison", "best_tool", "enterprise"):
            # Try to match partial
            for cat in ("beginner", "troubleshooting", "comparison", "best_tool", "enterprise"):
                if cat in category:
                    category = cat
                    break
            else:
                category = "general"

        variants.append(QueryVariant(
            seed_query=seed_query,
            variant_text=variant_text,
            category=category,
            similarity_to_seed=0.0,
        ))
    return variants


def _diversity_filter(
    variants: list[QueryVariant],
    emb: object,
    threshold: float,
) -> list[QueryVariant]:
    """Remove variants that are too similar to already-accepted variants."""
    if not variants:
        return []

    accepted = [variants[0]]
    for candidate in variants[1:]:
        if candidate.embedding is None:
            accepted.append(candidate)
            continue
        too_similar = False
        for existing in accepted:
            if existing.embedding is None:
                continue
            sim = emb.cosine_similarity(candidate.embedding, existing.embedding)
            if sim > threshold:
                logger.debug(
                    "variant_rejected_similarity",
                    candidate=candidate.variant_text[:50],
                    sim=round(sim, 3),
                )
                too_similar = True
                break
        if not too_similar:
            accepted.append(candidate)

    return accepted


def _get_expander_llm(
    cfg,
    preferred_provider: str | None = None,
    preferred_model_id: str | None = None,
):
    """Get the best available LLM for query expansion."""
    if preferred_model_id:
        preferred_model = next(
            (m for m in cfg.models.get_all() if m.model_id == preferred_model_id),
            None,
        )
        if preferred_model:
            if preferred_model.provider == "anthropic" and cfg.anthropic_api_key:
                from langchain_anthropic import ChatAnthropic
                logger.info("expander_llm", provider="anthropic", model=preferred_model.model_id)
                return ChatAnthropic(
                    model=preferred_model.model_id,
                    api_key=cfg.anthropic_api_key,
                    temperature=0.7,
                    max_tokens=512,
                )
            if preferred_model.provider == "google" and cfg.google_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info("expander_llm", provider="google", model=preferred_model.model_id)
                return ChatGoogleGenerativeAI(
                    model=preferred_model.model_id,
                    google_api_key=cfg.google_api_key,
                    temperature=0.7,
                    max_output_tokens=512,
                )

    provider_order = ["anthropic", "google"]
    if preferred_provider in provider_order:
        provider_order.remove(preferred_provider)
        provider_order.insert(0, preferred_provider)

    for provider in provider_order:
        if provider == "anthropic" and cfg.anthropic_api_key and cfg.models.claude:
            model = None
            for m in cfg.models.claude:
                if m.tier == "low":
                    model = m
                    break
            if not model:
                model = cfg.models.claude[0]
            from langchain_anthropic import ChatAnthropic
            logger.info("expander_llm", provider="anthropic", model=model.model_id)
            return ChatAnthropic(model=model.model_id, api_key=cfg.anthropic_api_key, temperature=0.7, max_tokens=512)

        if provider == "google" and cfg.google_api_key and cfg.models.gemini:
            model = None
            for m in cfg.models.gemini:
                if m.tier == "low":
                    model = m
                    break
            if not model:
                model = cfg.models.gemini[0]
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("expander_llm", provider="google", model=model.model_id)
            return ChatGoogleGenerativeAI(model=model.model_id, google_api_key=cfg.google_api_key, temperature=0.7, max_output_tokens=512)

    return None


def _fallback_expansion(
    seed_query: str,
    seed_embedding: list[float],
    emb: object,
) -> list[QueryVariant]:
    """Basic fallback when no LLM is available for expansion."""
    templates = {
        "beginner": f"how to get started with {seed_query}",
        "troubleshooting": f"troubleshooting {seed_query} not working",
        "comparison": f"best alternatives for {seed_query}",
        "best_tool": f"what is the best tool for {seed_query}",
        "enterprise": f"enterprise solution for {seed_query} at scale",
    }
    variants = []
    for category, text in templates.items():
        variants.append(QueryVariant(
            seed_query=seed_query,
            variant_text=text,
            category=category,
            similarity_to_seed=0.0,
        ))
    return variants
