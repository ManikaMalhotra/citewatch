"""
Citewatch — Main entry point.

Typer-based CLI with Rich output for all GEO analysis commands.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from citewatch.models import CitationStatus, GEOAnalysis

# Configure structlog for clean console output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()
console = Console()

app = typer.Typer(
    name="citewatch",
    help="🔍 Citewatch — Generative Engine Optimization for any brand",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


def _preferred_provider_for_models(resolved) -> Optional[str]:
    """Use the first selected model's provider to steer supporting LLM calls."""
    if not resolved:
        return None
    return resolved[0].provider


def _preferred_model_id_for_models(resolved) -> Optional[str]:
    """Use the first selected model's exact model id for supporting LLM calls."""
    if not resolved:
        return None
    return resolved[0].model_id


def _compact_error(exc: Exception, limit: int = 220) -> str:
    """Collapse noisy provider exceptions into a short single line."""
    message = " ".join(str(exc).split())
    if len(message) <= limit:
        return message
    return f"{message[:limit - 3]}..."


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection for provider quota / rate-limit failures."""
    message = str(exc).lower()
    needles = [
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "rate-limit",
        "too many requests",
        "429",
    ]
    return any(needle in message for needle in needles)


def _classify_error(exc: Exception) -> str:
    """Bucket common provider failures for reporting."""
    message = str(exc).lower()
    if _is_rate_limit_error(exc):
        return "rate_limit"
    if "context length" in message or "input length exceeds" in message:
        return "context_length"
    if "not_found_error" in message or "model:" in message:
        return "model_not_found"
    return "error"


def _make_failure(phase: str, model: str, query: str, exc: Exception) -> dict[str, str]:
    """Create a small structured error record for partial reports."""
    return {
        "phase": phase,
        "model": model,
        "query": query,
        "error_type": _classify_error(exc),
        "message": _compact_error(exc),
    }


def _build_partial_analysis(
    query: str,
    responses,
    reason: str,
    intended_models: list[str] | None = None,
) -> GEOAnalysis:
    """Create a reportable placeholder analysis when the judge cannot run."""
    models_analyzed = [r.model for r in responses] or (intended_models or [])
    return GEOAnalysis(
        query=query,
        models_analyzed=models_analyzed,
        citation_status=CitationStatus.ABSENT,
        brand_mentioned=False,
        brand_recommended=False,
        why_brand_missing=reason,
        confidence=0.0,
        raw_judge_reasoning="",
    )


def _kb_seed_query(cfg) -> str:
    """A short embedding query used to sample the knowledge base."""
    parts = [cfg.target.name, cfg.target.domain_context or cfg.target.description]
    return " ".join(p for p in parts if p).strip() or cfg.target.domain


def _analysis_parse_failed(analysis: GEOAnalysis) -> bool:
    """Detect the known empty-analysis fallback used after judge parse failures."""
    return (
        analysis.confidence == 0.0
        and analysis.why_brand_missing == "Analysis could not be completed — judge response parsing failed"
    )


# ── Commands ──────────────────────────────────────────────────────────────

@app.command()
def expand(
    query: str = typer.Argument(help="Seed query to expand"),
    num: int = typer.Option(5, "--num", "-n", help="Number of variants to generate"),
    threshold: float = typer.Option(0.85, "--threshold", "-t", help="Diversity threshold (0-1)"),
):
    """🧬 Expand a seed query into diverse semantic variants."""
    console.print(Panel(f"[bold cyan]Expanding query:[/] {query}", title="Query Expander"))

    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    from citewatch.embeddings import get_embedding_manager
    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Checking Ollama...", total=None)
        try:
            emb.health_check()
        except RuntimeError as e:
            console.print(f"[red]❌ {e}[/red]")
            raise typer.Exit(1)

        progress.add_task("Generating variants...", total=None)
        from citewatch.expander import expand_query
        variants = _run_async(expand_query(query, num_variants=num, diversity_threshold=threshold))

    # Display results
    table = Table(title="Query Variants", show_lines=True)
    table.add_column("Category", style="cyan", width=16)
    table.add_column("Variant", style="white")
    table.add_column("Similarity", style="yellow", justify="right", width=12)

    for v in variants:
        table.add_row(v.category, v.variant_text, f"{v.similarity_to_seed:.3f}")

    console.print(table)
    console.print(f"\n[green]✅ Generated {len(variants)} variants[/green]")


@app.command()
def capture(
    query: str = typer.Argument(help="Query to capture LLM responses for"),
    models: str = typer.Option("haiku", "--models", "-m", help="Comma-separated model aliases"),
    expand_first: bool = typer.Option(False, "--expand", "-e", help="Expand query into variants first"),
):
    """📸 Capture LLM responses for a query across models."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    model_aliases = [m.strip() for m in models.split(",")]
    resolved = cfg.models.resolve_model_ids(model_aliases)

    if not resolved:
        console.print(f"[red]❌ No valid models found for aliases: {models}[/red]")
        console.print(f"Available: {', '.join(m.alias for m in cfg.models.get_all())}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Query:[/] {query}\n[bold cyan]Models:[/] {', '.join(m.alias for m in resolved)}",
        title="LLM Capture",
    ))

    # Optionally expand first
    queries = [query]
    if expand_first:
        from citewatch.expander import expand_query
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Expanding query...", total=None)
            variants = _run_async(
                expand_query(
                    query,
                    preferred_provider=_preferred_provider_for_models(resolved),
                    preferred_model_id=_preferred_model_id_for_models(resolved),
                )
            )
        queries = [query] + [v.variant_text for v in variants]
        console.print(f"[cyan]📝 Will capture {len(queries)} queries × {len(resolved)} models = {len(queries) * len(resolved)} total[/cyan]")

    # Set up capture engines
    from citewatch.cache import CacheManager
    from citewatch.capture.anthropic import AnthropicCaptureEngine
    from citewatch.capture.google import GoogleCaptureEngine

    cache = CacheManager(db_path=cfg.cache.db_path, ttl_days=cfg.cache.ttl_days)
    engines = {}

    if cfg.anthropic_api_key:
        engines["anthropic"] = AnthropicCaptureEngine(
            api_key=cfg.anthropic_api_key,
            cache=cache,
            max_concurrent=cfg.capture.max_concurrent,
            retry_max_attempts=cfg.capture.retry_max_attempts,
            retry_base_delay=cfg.capture.retry_base_delay,
        )

    if cfg.google_api_key:
        engines["google"] = GoogleCaptureEngine(
            api_key=cfg.google_api_key,
            cache=cache,
            max_concurrent=cfg.capture.max_concurrent,
            retry_max_attempts=cfg.capture.retry_max_attempts,
            retry_base_delay=cfg.capture.retry_base_delay,
        )

    # Run captures
    all_responses = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for q in queries:
            for model in resolved:
                engine = engines.get(model.provider)
                if not engine:
                    console.print(f"[yellow]⚠️ No API key for {model.provider} — skipping {model.alias}[/yellow]")
                    continue

                task = progress.add_task(f"Capturing {model.alias}: {q[:40]}...", total=None)
                try:
                    resp = _run_async(engine.capture(
                        query=q,
                        model_id=model.model_id,
                        temperature=cfg.capture.temperature,
                        max_tokens=cfg.capture.max_tokens,
                    ))
                    all_responses.append(resp)
                    progress.remove_task(task)
                except Exception as e:
                    console.print(f"[red]❌ Failed {model.alias}: {e}[/red]")
                    progress.remove_task(task)

    # Display summary
    table = Table(title="Capture Results", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Query", style="white", max_width=40)
    table.add_column("Brands", style="yellow")
    table.add_column("Citations", style="green", justify="right")
    table.add_column("Tokens", style="dim", justify="right")

    for resp in all_responses:
        brands = ", ".join(b.brand for b in resp.mentioned_brands[:5])
        tokens = resp.token_usage.get("output_tokens", "?")
        table.add_row(
            resp.model.split("/")[-1],
            resp.query[:40],
            brands or "none",
            str(len(resp.citations)),
            str(tokens),
        )

    console.print(table)
    console.print(f"\n[green]✅ Captured {len(all_responses)} responses (cache: {cache.count} entries)[/green]")


@app.command()
def analyze(
    query: str = typer.Argument(help="Query to analyze"),
    models: str = typer.Option("haiku", "--models", "-m", help="Comma-separated model aliases"),
    compare: str = typer.Option("", "--compare", "-c", help="Comma-separated competitor names to compare"),
    skip_rate_limits: bool = typer.Option(
        True,
        "--skip-rate-limits/--fail-on-rate-limits",
        help="Continue and write a partial report when provider quotas/rate limits are hit",
    ),
):
    """🔬 Run GEO analysis on captured (or fresh) LLM responses."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    console.print(Panel(f"[bold cyan]Analyzing:[/] {query}", title="GEO Analyzer"))

    # Step 1: Capture responses (reuses cache if available)
    model_aliases = [m.strip() for m in models.split(",")]
    resolved = cfg.models.resolve_model_ids(model_aliases)

    from citewatch.cache import CacheManager, KnowledgeBase
    from citewatch.capture.anthropic import AnthropicCaptureEngine
    from citewatch.capture.google import GoogleCaptureEngine

    cache = CacheManager(db_path=cfg.cache.db_path, ttl_days=cfg.cache.ttl_days)
    kb = KnowledgeBase(db_path=cfg.ingestion.db_path)

    engines = {}
    if cfg.anthropic_api_key:
        engines["anthropic"] = AnthropicCaptureEngine(
            api_key=cfg.anthropic_api_key, cache=cache,
            max_concurrent=cfg.capture.max_concurrent,
        )
    if cfg.google_api_key:
        engines["google"] = GoogleCaptureEngine(
            api_key=cfg.google_api_key, cache=cache,
            max_concurrent=cfg.capture.max_concurrent,
        )

    responses = []
    capture_failures = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for model in resolved:
            engine = engines.get(model.provider)
            if not engine:
                continue
            task = progress.add_task(f"Capturing {model.alias}...", total=None)
            try:
                resp = _run_async(engine.capture(query=query, model_id=model.model_id))
                responses.append(resp)
            except Exception as e:
                failure = _make_failure("capture", model.alias, query, e)
                capture_failures.append(failure)
                console.print(f"[red]❌ {model.alias}: {failure['message']}[/red]")
            progress.remove_task(task)

    intended_models = [m.model_id for m in resolved]
    if not responses:
        reason = "No model responses were captured, so GEO analysis was skipped."
        from citewatch.analyzer.reporter import generate_geo_report
        analysis = _build_partial_analysis(query, responses, reason, intended_models=intended_models)
        paths = generate_geo_report(
            query,
            analysis,
            responses,
            output_dir=cfg.reports.output_dir,
            report_status="partial",
            run_notes=[reason],
            capture_failures=capture_failures,
            analysis_error=reason,
            intended_models=intended_models,
        )
        console.print("[yellow]⚠️ No responses captured — wrote a partial report instead[/yellow]")
        console.print(f"  Markdown: {paths['markdown']}")
        console.print(f"  JSON: {paths['json']}")
        return

    # Step 2: Run GEO analysis
    analysis_error = ""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running GEO analysis...", total=None)

        compare_brands = [b.strip() for b in compare.split(",") if b.strip()] or None
        from citewatch.analyzer.geo import analyze_query
        try:
            analysis = _run_async(
                analyze_query(
                    query,
                    responses,
                    kb,
                    compare_brands,
                    preferred_provider=_preferred_provider_for_models(resolved),
                    preferred_model_id=_preferred_model_id_for_models(resolved),
                )
            )
            if _analysis_parse_failed(analysis):
                analysis_error = analysis.why_brand_missing
        except Exception as e:
            analysis_error = _compact_error(e)
            analysis = _build_partial_analysis(
                query,
                responses,
                f"GEO judge did not complete: {analysis_error}",
                intended_models=intended_models,
            )
            from citewatch.analyzer.reporter import generate_geo_report
            paths = generate_geo_report(
                query,
                analysis,
                responses,
                output_dir=cfg.reports.output_dir,
                report_status="partial",
                run_notes=["The GEO judge failed before a full analysis could be produced."],
                capture_failures=capture_failures,
                analysis_error=analysis_error,
                intended_models=intended_models,
            )
            if skip_rate_limits and _is_rate_limit_error(e):
                console.print("[yellow]⚠️ Rate limit hit — wrote a partial report instead[/yellow]")
                console.print(f"  Markdown: {paths['markdown']}")
                console.print(f"  JSON: {paths['json']}")
                return
            console.print("[yellow]⚠️ Wrote a partial report before stopping[/yellow]")
            console.print(f"  Markdown: {paths['markdown']}")
            console.print(f"  JSON: {paths['json']}")
            raise typer.Exit(1) from e

    # Step 3: Generate report
    from citewatch.analyzer.reporter import generate_geo_report
    report_status = "partial" if analysis_error else "complete"
    run_notes = []
    if analysis_error:
        run_notes.append("The GEO analysis did not fully complete, so this is a partial report.")
    paths = generate_geo_report(
        query,
        analysis,
        responses,
        output_dir=cfg.reports.output_dir,
        report_status=report_status,
        run_notes=run_notes,
        capture_failures=capture_failures,
        analysis_error=analysis_error,
        intended_models=intended_models,
    )

    # Display summary
    console.print(Panel(
        f"[bold]Citation Status:[/] {analysis.citation_status.value.upper()}\n"
        f"[bold]Brand Mentioned:[/] {'✅ Yes' if analysis.brand_mentioned else '❌ No'}\n"
        f"[bold]Brand Recommended:[/] {'⭐ Yes' if analysis.brand_recommended else '— No'}\n"
        f"[bold]Confidence:[/] {analysis.confidence:.0%}\n"
        f"[bold]Fixes:[/] {len(analysis.recommended_fixes)} recommendations",
        title="🔬 GEO Analysis Result",
    ))

    if analysis.recommended_fixes:
        table = Table(title="Top Fixes", show_lines=True)
        table.add_column("Priority", style="bold", width=10)
        table.add_column("Action", style="white")
        table.add_column("Impact", style="cyan", max_width=40)

        for fix in analysis.recommended_fixes[:5]:
            table.add_row(fix.priority.value.upper(), fix.action[:60], fix.expected_impact[:40])
        console.print(table)

    console.print(f"\n[green]📄 Report saved:[/green]")
    console.print(f"  Markdown: {paths['markdown']}")
    console.print(f"  JSON: {paths['json']}")


@app.command()
def ingest(
    site: Optional[str] = typer.Option(None, "--site", "-s", help="Site to crawl (defaults to target.domain in config.yaml)"),
    max_depth: int = typer.Option(4, "--max-depth", "-d", help="Max URL path depth"),
    update: bool = typer.Option(False, "--update", "-u", help="Incremental update mode"),
):
    """📥 Ingest the target site into the knowledge base."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()
    site = (site or f"https://{cfg.target.domain}").rstrip("/")

    console.print(Panel(f"[bold cyan]Ingesting:[/] {site}", title="Content Ingestion"))

    from citewatch.cache import KnowledgeBase
    from citewatch.embeddings import get_embedding_manager
    from citewatch.ingestion.sitemap import fetch_sitemap_urls, filter_urls_by_depth
    from citewatch.ingestion.crawler import crawl_urls
    from citewatch.ingestion.chunker import chunk_page

    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)
    kb = KnowledgeBase(db_path=cfg.ingestion.db_path, embedding_mgr=emb)

    # Step 1: Fetch sitemap
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Fetching sitemap...", total=None)
        site_host = site.replace("https://", "").replace("http://", "").split("/")[0]
        if cfg.target.domain in site_host:
            sitemap_url = cfg.target.sitemap_url
        else:
            sitemap_url = f"{site}/sitemap.xml"
        entries = _run_async(fetch_sitemap_urls(sitemap_url, max_depth=max_depth))

    if not entries:
        console.print("[red]❌ No URLs found in sitemap[/red]")
        raise typer.Exit(1)

    # Filter by depth
    entries = filter_urls_by_depth(entries, max_depth)
    console.print(f"[cyan]📝 Found {len(entries)} URLs (depth ≤ {max_depth})[/cyan]")

    # Step 2: Crawl pages
    urls = [e["url"] for e in entries]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Crawling {len(urls)} pages...", total=None)
        pages = _run_async(crawl_urls(
            urls,
            max_concurrent=cfg.ingestion.max_concurrent_crawl,
            request_delay=cfg.ingestion.request_delay,
            user_agent=cfg.ingestion.user_agent,
        ))

    console.print(f"[cyan]✅ Crawled {len(pages)} pages successfully[/cyan]")

    # Step 3: Chunk and ingest
    total_chunks = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Chunking and embedding...", total=None)
        for page in pages:
            if not page.body_text:
                continue
            chunks = chunk_page(
                url=page.url,
                text=page.body_text,
                headings=page.headings,
                chunk_size=cfg.ingestion.chunk_size,
                overlap=cfg.ingestion.chunk_overlap,
            )
            if chunks:
                kb.add_chunks(
                    chunk_ids=[c.chunk_id for c in chunks],
                    documents=[c.text for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                )
                total_chunks += len(chunks)

    console.print(Panel(
        f"[bold green]✅ Ingestion complete[/bold green]\n"
        f"Pages crawled: {len(pages)}\n"
        f"Chunks stored: {total_chunks}\n"
        f"KB total: {kb.count} chunks",
        title="Ingestion Summary",
    ))


@app.command(name="ingest-competitor")
def ingest_competitor(
    competitors: str = typer.Option("", "--competitors", "-c", help="Comma-separated competitor domains (overrides config.yaml)"),
    max_depth: int = typer.Option(3, "--max-depth", "-d", help="Max URL path depth to crawl"),
    max_urls: int = typer.Option(100, "--max-urls", "-n", help="Max URLs to crawl per competitor domain"),
    request_delay: float = typer.Option(1.0, "--delay", help="Delay between requests in seconds (polite crawling)"),
):
    """📥 Ingest competitor site content into the competitor knowledge base (like ingest, but for competitors)."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    # Determine competitor domains
    comp_domains = [c.strip() for c in competitors.split(",") if c.strip()] if competitors else [
        c.domain for c in cfg.competitors
    ]

    if not comp_domains:
        console.print("[red]❌ No competitor domains. Add them to config.yaml or use --competitors[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Competitor Ingestion[/bold cyan]\n"
        f"Domains: {', '.join(comp_domains)}\n"
        f"Max depth: {max_depth}\n"
        f"Max URLs per domain: {max_urls}",
        title="📥 Competitor Ingestion",
    ))

    # Initialize embeddings
    from citewatch.embeddings import get_embedding_manager
    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)
    try:
        emb.health_check()
    except RuntimeError as e:
        console.print(f"[red]❌ Embeddings not available: {e}[/red]")
        raise typer.Exit(1)

    from citewatch.audit.competitor import discover_competitor_urls, crawl_competitor_content, CompetitorKB
    from citewatch.ingestion.chunker import chunk_page

    # Step 1: Discover URLs
    console.print("\n[bold]━━━ Step 1: Discover Competitor URLs ━━━[/bold]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Discovering URLs from {len(comp_domains)} competitor sitemaps...", total=None)
        domain_urls = _run_async(discover_competitor_urls(
            comp_domains,
            max_urls_per_domain=max_urls,
            max_depth=max_depth,
        ))

    total_urls = sum(len(urls) for urls in domain_urls.values())
    console.print(f"[cyan]📝 Found {total_urls} competitor URLs[/cyan]")
    for domain, urls in domain_urls.items():
        console.print(f"  • {domain}: {len(urls)} URLs")

    if total_urls == 0:
        console.print("[yellow]⚠️ No URLs discovered — check competitor domains and sitemap availability[/yellow]")
        raise typer.Exit(1)

    # Step 2: Crawl pages
    console.print("\n[bold]━━━ Step 2: Crawl Competitor Pages ━━━[/bold]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Crawling {total_urls} pages (delay: {request_delay}s)...", total=None)
        competitor_pages = _run_async(crawl_competitor_content(
            domain_urls,
            max_concurrent=3,
            request_delay=request_delay,
        ))

    console.print(f"[green]✅ Crawled {len(competitor_pages)} pages[/green]")

    # Step 3: Chunk and ingest into competitor KB
    console.print("\n[bold]━━━ Step 3: Chunk & Embed ━━━[/bold]")

    comp_kb = CompetitorKB(db_path=cfg.ingestion.competitor_db_path, embedding_mgr=emb)
    total_chunks = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Chunking and embedding competitor content...", total=None)

        # Also store page-level entries for page-level queries
        comp_kb.add_pages(competitor_pages)

        # Chunk each page and store chunks
        for page in competitor_pages:
            if not page.body_text:
                continue
            chunks = chunk_page(
                url=page.url,
                text=page.body_text,
                headings=page.headings,
                chunk_size=cfg.ingestion.chunk_size,
                overlap=cfg.ingestion.chunk_overlap,
            )
            if chunks:
                chunk_metadatas = []
                for c in chunks:
                    meta = c.metadata.copy()
                    meta["domain"] = page.domain
                    meta["type"] = "chunk"
                    meta["word_count"] = c.token_count * 4
                    meta["has_schema"] = str(page.has_schema_markup)
                    chunk_metadatas.append(meta)

                comp_kb.add_chunks(
                    chunk_ids=[c.chunk_id for c in chunks],
                    documents=[c.text for c in chunks],
                    metadatas=chunk_metadatas,
                )
                total_chunks += len(chunks)

    # Summary
    table = Table(title="Competitor Ingestion Summary", show_lines=True)
    table.add_column("Domain", style="cyan")
    table.add_column("Pages", justify="right", width=8)
    table.add_column("Status", width=10)

    for domain in comp_domains:
        domain_pages = [p for p in competitor_pages if p.domain == domain]
        status = "[green]✅[/green]" if domain_pages else "[red]❌[/red]"
        table.add_row(domain, str(len(domain_pages)), status)
    console.print(table)

    console.print(Panel(
        f"[bold green]✅ Competitor Ingestion Complete[/bold green]\n"
        f"Domains: {len(comp_domains)}\n"
        f"Pages crawled: {len(competitor_pages)}\n"
        f"Chunks stored: {total_chunks}\n"
        f"Competitor KB total: {comp_kb.count} entries",
        title="Ingestion Summary",
    ))


if __name__ == "__main__":
    app()
