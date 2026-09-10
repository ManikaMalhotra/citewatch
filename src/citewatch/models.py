"""
Pydantic v2 data models for the Citewatch pipeline.

These models are the single source of truth for data shapes across
query expansion, LLM capture, GEO analysis, ingestion, and SEO audit.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class CitationStatus(str, Enum):
    """Whether the target brand was cited in an LLM response."""
    PRESENT = "present"
    ABSENT = "absent"
    PARTIAL = "partial"  # mentioned but not linked / recommended


class FixPriority(str, Enum):
    """Priority level for a recommended content fix."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Query Expansion ───────────────────────────────────────────────────────

class QueryVariant(BaseModel):
    """A semantically expanded variant of a seed query."""
    seed_query: str = Field(description="The original seed query")
    variant_text: str = Field(description="The expanded variant")
    category: str = Field(description="Intent category: beginner, troubleshooting, comparison, best_tool, enterprise")
    similarity_to_seed: float = Field(ge=0.0, le=1.0, description="Cosine similarity to the seed query embedding")
    embedding: list[float] | None = Field(default=None, exclude=True, description="Raw embedding vector (not serialized)")


# ── LLM Capture ───────────────────────────────────────────────────────────

class CoTStep(BaseModel):
    """A single chain-of-thought reasoning step from an LLM response."""
    step: int
    thinking: str


class Citation(BaseModel):
    """A citation/source reference extracted from an LLM response."""
    url: str = Field(default="", description="URL if provided")
    title: str = Field(default="", description="Title or label of the source")
    snippet: str = Field(default="", description="Relevant text snippet")


class BrandMention(BaseModel):
    """A brand/product mention extracted from an LLM response."""
    brand: str = Field(description="Normalized brand name (lowercase)")
    context: str = Field(description="The surrounding text where the brand was mentioned")
    sentiment: Literal["positive", "neutral", "negative"] = Field(default="neutral")
    is_recommended: bool = Field(default=False, description="Whether the LLM explicitly recommends this brand")
    reason: str = Field(default="", description="Why the LLM mentioned this brand")


class LLMResponse(BaseModel):
    """Structured representation of a captured LLM response."""
    model: str = Field(description="Model identifier used for the capture")
    provider: str = Field(description="Provider: anthropic or google")
    query: str = Field(description="The query that was sent")
    variant_category: str = Field(default="original", description="Which expansion category this query came from")
    mode: Literal["api"] = Field(default="api", description="Capture mode (API only in MVP)")
    raw_text: str = Field(description="Full raw text response from the LLM")
    cot_steps: list[CoTStep] = Field(default_factory=list, description="Extracted chain-of-thought steps")
    citations: list[Citation] = Field(default_factory=list, description="Extracted citations/sources")
    mentioned_brands: list[BrandMention] = Field(default_factory=list, description="All brand mentions detected")
    sources: list[str] = Field(default_factory=list, description="Raw source URLs mentioned")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_usage: dict[str, int] = Field(default_factory=dict, description="Token usage: input_tokens, output_tokens")
    cache_key: str = Field(default="", description="SHA256 cache key for this capture")


# ── GEO Analysis ──────────────────────────────────────────────────────────

class RecommendedFix(BaseModel):
    """A single actionable fix to improve GEO performance."""
    priority: FixPriority
    action: str = Field(description="What to do (specific, actionable)")
    target_page: str = Field(default="", description="Which page/URL to modify")
    expected_impact: str = Field(description="Why this will improve citation likelihood")
    effort: Literal["low", "medium", "high"] = Field(default="medium")


class CompetitorSignal(BaseModel):
    """Why a competitor was cited by the LLM."""
    brand: str
    signal_type: str = Field(description="e.g., 'E-E-A-T', 'recency', 'schema_markup', 'backlinks', 'exact_phrase_match'")
    detail: str = Field(description="Specific explanation")
    strength: Literal["strong", "moderate", "weak"] = Field(default="moderate")


class GEOAnalysis(BaseModel):
    """Complete GEO analysis result for a single query."""
    query: str
    models_analyzed: list[str] = Field(description="Which models were queried")
    citation_status: CitationStatus
    brand_mentioned: bool = Field(default=False, description="Whether the configured target brand was mentioned")
    brand_recommended: bool = Field(default=False, description="Whether the configured target brand was recommended")
    brand_context: str = Field(default="", description="How the target brand was described (if mentioned)")
    why_brand_missing: str = Field(default="", description="Analysis of why the target brand was not cited")
    competitor_signals: list[CompetitorSignal] = Field(default_factory=list)
    authority_signals_missing: list[str] = Field(default_factory=list, description="What the target brand lacks vs. competitors")
    recommended_fixes: list[RecommendedFix] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this analysis")
    raw_judge_reasoning: str = Field(default="", description="Full reasoning from the GEO judge LLM")


# ── Ingestion ─────────────────────────────────────────────────────────────

class CrawledPage(BaseModel):
    """Metadata for a single crawled page from the target site."""
    url: str
    title: str = ""
    meta_description: str = ""
    headings: list[str] = Field(default_factory=list, description="All headings (h1-h6)")
    body_text: str = Field(default="", description="Cleaned body text")
    last_modified: str = ""
    etag: str = ""
    status_code: int = 200
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: int = 0
    has_schema_markup: bool = False
    schema_types: list[str] = Field(default_factory=list, description="e.g., FAQPage, HowTo, Article")


class ContentChunk(BaseModel):
    """A semantic chunk of content from a crawled page."""
    chunk_id: str = Field(description="Unique identifier for this chunk")
    source_url: str
    text: str
    heading_context: str = Field(default="", description="The heading this chunk falls under")
    position: int = Field(default=0, description="Position of this chunk within the page")
    token_count: int = 0
    metadata: dict = Field(default_factory=dict)


# ── SEO Audit ─────────────────────────────────────────────────────────────

class CannibalizationCluster(BaseModel):
    """A group of pages that cannibalize each other for similar queries."""
    queries: list[str]
    pages: list[str] = Field(description="URLs of competing pages")
    max_similarity: float
    recommendation: str = ""


class AIContentScore(BaseModel):
    """AI-generated content detection result for a page."""
    url: str
    entity_density: float = Field(description="Named entities per sentence")
    sentence_length_variance: float = Field(description="Std dev of sentence lengths")
    vocabulary_richness: float = Field(description="Type-token ratio")
    overall_score: float = Field(ge=0.0, le=1.0, description="0=human, 1=likely AI")
    verdict: Literal["human", "mixed", "likely_ai"] = "human"
    signals: list[str] = Field(default_factory=list, description="Which signals triggered")


class SEOAuditResult(BaseModel):
    """Complete SEO audit result."""
    total_pages_analyzed: int = 0
    cannibalization_clusters: list[CannibalizationCluster] = Field(default_factory=list)
    ai_content_scores: list[AIContentScore] = Field(default_factory=list)
    missing_schema_pages: list[str] = Field(default_factory=list)
    duplicate_titles: list[dict] = Field(default_factory=list)
    eeat_gaps: list[dict] = Field(default_factory=list)
    top_recommendations: list[RecommendedFix] = Field(default_factory=list)


# ── Competitive SEO Analysis ──────────────────────────────────────────────

class KeywordEntry(BaseModel):
    """A keyword from GSC/Ahrefs with performance data."""
    keyword: str
    url: str = Field(default="", description="Page ranking for this keyword (if any)")
    position: float = Field(default=0.0, description="Current avg position (0 = not ranking)")
    clicks: int = 0
    impressions: int = 0
    ctr: float = 0.0
    search_volume: int = 0
    difficulty: float = 0.0
    source: str = Field(default="", description="gsc, ahrefs, or competitor")


class KeywordGap(BaseModel):
    """A keyword we should rank for but don't (or rank poorly)."""
    keyword: str
    competitor_url: str = Field(description="Competitor page ranking for it")
    competitor_domain: str = ""
    competitor_position: float = 0.0
    our_closest_page: str = Field(default="", description="Our most relevant page (by embedding similarity)")
    our_position: float = Field(default=0.0, description="Our current position, 0 = not ranking")
    difficulty: str = Field(default="medium", description="low/medium/high")
    search_volume: int = 0
    opportunity_score: float = Field(default=0.0, description="0-1 score combining volume, difficulty, relevance")
    recommendation: str = ""


class KeywordCluster(BaseModel):
    """A group of semantically similar keywords targeting the same intent."""
    cluster_id: str
    primary_keyword: str = Field(description="The highest-volume keyword in the cluster")
    keywords: list[str]
    our_pages: list[str] = Field(default_factory=list, description="Our URLs targeting this cluster")
    competitor_pages: list[str] = Field(default_factory=list, description="Competitor URLs in this cluster")
    avg_position: float = 0.0
    total_volume: int = 0
    is_cannibalized: bool = Field(default=False, description="True if multiple of our pages target this cluster")


class CompetitorPage(BaseModel):
    """A competitor article with content for comparison."""
    url: str
    domain: str
    title: str = ""
    meta_description: str = ""
    body_text: str = Field(default="", description="Cleaned body text")
    word_count: int = 0
    headings: list[str] = Field(default_factory=list)
    target_keywords: list[str] = Field(default_factory=list)
    has_schema_markup: bool = False
    crawled_at: datetime = Field(default_factory=datetime.utcnow)


class ArticleComparison(BaseModel):
    """Head-to-head comparison of our page vs a competitor's."""
    keyword_cluster: str = Field(description="The keyword cluster this comparison is about")
    our_url: str
    our_title: str = ""
    competitor_url: str
    competitor_domain: str = ""
    competitor_title: str = ""
    shared_keywords: list[str] = Field(default_factory=list)
    content_similarity: float = Field(default=0.0, description="Embedding cosine similarity")
    word_count_ours: int = 0
    word_count_theirs: int = 0
    strengths_theirs: list[str] = Field(default_factory=list, description="What the competitor does better")
    gaps_ours: list[str] = Field(default_factory=list, description="What our content is missing")
    recommended_actions: list[str] = Field(default_factory=list, description="Specific improvement steps")
    raw_analysis: str = Field(default="", description="Full LLM analysis text")


class CompetitiveAuditResult(BaseModel):
    """Full competitive SEO audit result."""
    keyword_map: list[KeywordEntry] = Field(default_factory=list, description="Our current keyword coverage")
    keyword_gaps: list[KeywordGap] = Field(default_factory=list, description="Keywords competitors rank for that we don't")
    keyword_clusters: list[KeywordCluster] = Field(default_factory=list)
    competitor_articles: list[CompetitorPage] = Field(default_factory=list)
    article_comparisons: list[ArticleComparison] = Field(default_factory=list)
    cannibalization_clusters: list[CannibalizationCluster] = Field(default_factory=list)
    top_recommendations: list[RecommendedFix] = Field(default_factory=list)
    domains_analyzed: list[str] = Field(default_factory=list)
    total_keywords_analyzed: int = 0

