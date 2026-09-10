"""
Configuration manager for Citewatch.

Merges environment variables (.env) + config.yaml + CLI overrides.
Validates all configuration at startup via Pydantic Settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger()

# ── Nested Config Models ──────────────────────────────────────────────────

class TargetConfig(BaseModel):
    name: str = "your-brand"
    domain: str = "example.com"
    sitemap_url: str = "https://example.com/sitemap.xml"
    max_crawl_depth: int = 4
    description: str = "Describe your product in one sentence — used by the GEO judge"
    domain_context: str = "the industry and topics your product sits in"


class CompetitorConfig(BaseModel):
    name: str
    domain: str


class ModelEntry(BaseModel):
    alias: str
    model_id: str
    provider: str  # "anthropic" or "google"
    tier: str = "mid"  # "high", "mid", "low"


class ModelsConfig(BaseModel):
    claude: list[ModelEntry] = Field(default_factory=list)
    gemini: list[ModelEntry] = Field(default_factory=list)

    def get_all(self) -> list[ModelEntry]:
        """Return all model entries across all providers."""
        return self.claude + self.gemini

    def get_by_alias(self, alias: str) -> ModelEntry | None:
        """Look up a model by its alias."""
        for entry in self.get_all():
            if entry.alias == alias:
                return entry
        return None

    def get_by_provider(self, provider: str) -> list[ModelEntry]:
        """Get all models for a specific provider."""
        if provider == "anthropic":
            return self.claude
        elif provider == "google":
            return self.gemini
        return []

    def resolve_model_ids(self, aliases: list[str]) -> list[ModelEntry]:
        """Resolve a list of aliases to ModelEntry objects."""
        resolved = []
        for alias in aliases:
            entry = self.get_by_alias(alias)
            if entry:
                resolved.append(entry)
            else:
                logger.warning("unknown_model_alias", alias=alias)
        return resolved


class ExpanderConfig(BaseModel):
    num_variants: int = 5
    diversity_threshold: float = 0.85
    categories: list[str] = Field(
        default_factory=lambda: ["beginner", "troubleshooting", "comparison", "best_tool", "enterprise"]
    )


class CaptureConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 4096
    max_concurrent: int = 3
    retry_max_attempts: int = 3
    retry_base_delay: float = 2.0


class CacheConfig(BaseModel):
    ttl_days: int = 30
    db_path: str = "./.citewatch_cache"


class IngestionConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: float = 0.2
    dedup_threshold: float = 0.92
    db_path: str = "./.citewatch_kb"
    competitor_db_path: str = "./.citewatch_competitor_kb"
    max_concurrent_crawl: int = 5
    request_delay: float = 0.5
    user_agent: str = "citewatch/0.1.0 (+https://github.com/ManikaMalhotra/citewatch)"


class AuditConfig(BaseModel):
    cannibalization_threshold: float = 0.82
    ai_gen_entity_density_min: float = 0.03
    coverage_topics: list[str] = Field(default_factory=list)


class BrandConfig(BaseModel):
    """Extra brand names to detect in LLM responses (in addition to target + competitors)."""
    detect: list[str] = Field(default_factory=list)


class ReportsConfig(BaseModel):
    output_dir: str = "./reports"
    formats: list[str] = Field(default_factory=lambda: ["markdown", "json"])


# ── Main Settings ─────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Top-level settings. Merges:
    1. .env file (ANTHROPIC_API_KEY, GOOGLE_API_KEY, OLLAMA_BASE_URL)
    2. config.yaml (everything else)
    3. CLI overrides (via runtime patching)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets from .env
    anthropic_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Structured config (loaded from YAML)
    target: TargetConfig = Field(default_factory=TargetConfig)
    competitors: list[CompetitorConfig] = Field(default_factory=list)
    brands: BrandConfig = Field(default_factory=BrandConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    expander: ExpanderConfig = Field(default_factory=ExpanderConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)

    def get_competitor_names(self) -> list[str]:
        """Return all competitor brand names (lowercase)."""
        return [c.name.lower() for c in self.competitors]

    def get_detectable_brands(self) -> list[str]:
        """Return brand names to scan for in captured LLM responses."""
        names = [self.target.name.lower()]
        names.extend(self.get_competitor_names())
        names.extend(b.lower().strip() for b in self.brands.detect if b.strip())
        # Preserve order, drop empties/dupes
        seen: set[str] = set()
        unique: list[str] = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique.append(name)
        return unique


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML config file contents."""
    if config_path is None:
        config_path = os.environ.get("CITEWATCH_CONFIG", "./config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning("config_yaml_not_found", path=str(config_path))
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    logger.info("config_loaded", path=str(config_path), keys=list(data.keys()))
    return data


def get_settings(config_path: str | Path | None = None) -> Settings:
    """
    Create and return a fully-merged Settings object.

    Priority: .env secrets → config.yaml → defaults
    """
    yaml_data = load_yaml_config(config_path)

    # Build settings: .env is auto-loaded by pydantic-settings,
    # YAML data is passed as init kwargs (overriding defaults but not env vars)
    settings = Settings(**yaml_data)

    # Sanitize: treat placeholder/template API keys as empty
    if _is_placeholder(settings.anthropic_api_key):
        settings.anthropic_api_key = ""
    if _is_placeholder(settings.google_api_key):
        settings.google_api_key = ""

    # Validate critical settings
    if not settings.anthropic_api_key:
        logger.warning("no_anthropic_key", msg="ANTHROPIC_API_KEY not set — Claude captures will fail")
    if not settings.google_api_key:
        logger.info("no_google_key", msg="GOOGLE_API_KEY not set — Gemini captures disabled")

    return settings


def _is_placeholder(value: str) -> bool:
    """Check if an API key is a placeholder/template value."""
    if not value:
        return True
    placeholders = ["xxxxx", "XXXXX", "your_", "YOUR_", "sk-ant-xxxxx", "AIzaXXXX", "<", "placeholder"]
    return any(p in value for p in placeholders)


# Module-level singleton (lazy-loaded)
_settings: Settings | None = None


def settings() -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (for testing)."""
    global _settings
    _settings = None
