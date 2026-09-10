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


@app.command(name="domain-authority")
def domain_authority(
    competitors: str = typer.Option("", "--competitors", "-c", help="Comma-separated competitor domains (overrides config)"),
):
    """📊 Compare domain authority signals between the target site and competitors."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    console.print(Panel("[bold cyan]Domain Authority Comparison[/]", title="📊 Domain Authority"))

    comp_domains = [c.strip() for c in competitors.split(",") if c.strip()] if competitors else [
        c.domain for c in cfg.competitors
    ]

    # Initialize
    from citewatch.embeddings import get_embedding_manager
    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)

    # Our KB stats
    from citewatch.cache import KnowledgeBase
    kb = KnowledgeBase(db_path=cfg.ingestion.db_path)

    our_chunks = kb.count
    our_stats = {
        "domain": cfg.target.domain,
        "chunks": our_chunks,
        "pages": 0,
        "total_words": 0,
        "avg_words_per_page": 0,
        "schema_count": 0,
        "schema_pct": 0.0,
    }

    # Try to compute our stats from KB metadata
    if our_chunks > 0:
        try:
            # Sample KB to estimate page count and word counts
            results = kb.query(_kb_seed_query(cfg), n_results=min(our_chunks, 200))
            if results and results.get("metadatas"):
                pages = set()
                total_words = 0
                for meta in results["metadatas"][0]:
                    url = meta.get("url", "")
                    if url:
                        pages.add(url)
                our_stats["pages"] = len(pages)
        except Exception:
            pass

    # Competitor KB stats
    from citewatch.audit.competitor import CompetitorKB
    comp_kb = CompetitorKB(db_path=cfg.ingestion.competitor_db_path, embedding_mgr=emb)

    if comp_kb.count == 0:
        console.print("[yellow]⚠️ Competitor KB is empty — run 'citewatch ingest-competitor' first[/yellow]")
        console.print(f"[cyan]📚 Our KB ({cfg.target.domain}): {our_chunks} chunks[/cyan]")
        raise typer.Exit(1)

    # Build comparison table
    console.print("\n[bold]━━━ Content Volume Comparison ━━━[/bold]")

    table = Table(title="Domain Authority Signals", show_lines=True)
    table.add_column("Domain", style="cyan", width=20)
    table.add_column("Pages", justify="right", width=8)
    table.add_column("Chunks", justify="right", width=8)
    table.add_column("Total Words", justify="right", width=12)
    table.add_column("Avg Words/Page", justify="right", width=14)
    table.add_column("Schema %", justify="right", width=10)

    # Our row
    table.add_row(
        f"[bold green]{our_stats['domain']}[/bold green] (us)",
        str(our_stats["pages"]),
        str(our_stats["chunks"]),
        str(our_stats["total_words"]),
        str(our_stats["avg_words_per_page"]),
        f"{our_stats['schema_pct']}%",
    )

    # Competitor rows
    all_comp_stats = []
    for domain in comp_domains:
        stats = comp_kb.get_domain_stats(domain)
        all_comp_stats.append(stats)
        if stats["pages"] > 0:
            table.add_row(
                stats["domain"],
                str(stats["pages"]),
                str(stats["chunks"]),
                str(stats["total_words"]),
                str(stats.get("avg_words_per_page", 0)),
                f"{stats.get('schema_pct', 0)}%",
            )

    console.print(table)

    # Topic coverage comparison
    console.print("\n[bold]━━━ Topic Coverage Comparison ━━━[/bold]")

    topics = cfg.audit.coverage_topics or [
        t.strip() for t in (cfg.target.domain_context or cfg.target.name).split(",") if t.strip()
    ]

    topic_table = Table(title="Topic Coverage (by relevance score)", show_lines=True)
    topic_table.add_column("Topic", style="cyan", width=30)
    topic_table.add_column(f"{cfg.target.domain}", justify="center", width=12)
    for domain in comp_domains[:5]:  # Top 5 competitors
        topic_table.add_column(domain[:15], justify="center", width=12)

    for topic in topics:
        row = [topic]

        # Our coverage
        try:
            our_results = kb.query(topic, n_results=3)
            our_docs = our_results.get("documents", [[]])[0] if our_results else []
            our_coverage = "✅ Strong" if len(our_docs) >= 2 else ("⚠️ Weak" if our_docs else "❌ None")
        except Exception:
            our_coverage = "❓"
        row.append(our_coverage)

        # Competitor coverage (top 5)
        for domain in comp_domains[:5]:
            try:
                comp_results = comp_kb.find_similar(topic, n_results=3, domain=domain)
                comp_docs = comp_results.get("documents", [[]])[0] if comp_results else []
                comp_coverage = "✅ Strong" if len(comp_docs) >= 2 else ("⚠️ Weak" if comp_docs else "❌ None")
            except Exception:
                comp_coverage = "❓"
            row.append(comp_coverage)

        topic_table.add_row(*row)

    console.print(topic_table)

    # Generate a report
    console.print("\n[bold]━━━ Recommendations ━━━[/bold]")

    recommendations = []
    for stats in all_comp_stats:
        if stats["pages"] > 0 and stats.get("avg_words_per_page", 0) > our_stats.get("avg_words_per_page", 0) * 1.5:
            recommendations.append(
                f"📝 {stats['domain']} has {stats.get('avg_words_per_page', 0)} avg words/page vs our "
                f"{our_stats.get('avg_words_per_page', 0)} — consider deepening content."
            )
        if stats.get("schema_pct", 0) > our_stats.get("schema_pct", 0) + 20:
            recommendations.append(
                f"🏷️ {stats['domain']} has {stats.get('schema_pct', 0)}% schema markup coverage vs our "
                f"{our_stats.get('schema_pct', 0)}% — add structured data."
            )
        if stats["pages"] > our_stats["pages"] * 2 and our_stats["pages"] > 0:
            recommendations.append(
                f"📄 {stats['domain']} has {stats['pages']} pages vs our {our_stats['pages']} — "
                f"consider expanding content library."
            )

    if recommendations:
        for rec in recommendations[:10]:
            console.print(f"  {rec}")
    else:
        console.print("  [green]✅ No major gaps detected based on available data[/green]")

    console.print(f"\n[dim]Tip: Run 'citewatch competitive --crawl-competitors' for full article-level analysis[/dim]")


@app.command()
def audit(
    gsc_csv: Optional[str] = typer.Option(None, "--gsc-csv", help="Path to GSC CSV export"),
    ahrefs_csv: Optional[str] = typer.Option(None, "--ahrefs-csv", help="Path to Ahrefs CSV export"),
):
    """🔎 Run SEO audit on ingested content."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    console.print(Panel("[bold cyan]Running SEO Audit[/]", title="SEO Audit"))

    from citewatch.cache import KnowledgeBase

    kb = KnowledgeBase(db_path=cfg.ingestion.db_path)
    if kb.count == 0:
        console.print("[yellow]⚠️ Knowledge base is empty — run 'citewatch ingest' first[/yellow]")
        console.print("[yellow]Running audit with limited data...[/yellow]")

    # We need crawled page data — re-derive from KB or crawl
    # For now, use empty pages list if KB only has chunks
    pages = []

    from citewatch.audit.seo import run_seo_audit
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Running audit...", total=None)
        result = _run_async(run_seo_audit(
            pages=pages,
            kb=kb,
            gsc_csv=gsc_csv,
            ahrefs_csv=ahrefs_csv,
            cannibalization_threshold=cfg.audit.cannibalization_threshold,
        ))

    from citewatch.analyzer.reporter import generate_audit_report
    paths = generate_audit_report(result, output_dir=cfg.reports.output_dir)

    console.print(Panel(
        f"[bold green]✅ Audit complete[/bold green]\n"
        f"Pages analyzed: {result.total_pages_analyzed}\n"
        f"Cannibalization clusters: {len(result.cannibalization_clusters)}\n"
        f"AI content flags: {len([s for s in result.ai_content_scores if s.verdict != 'human'])}\n"
        f"Missing schema: {len(result.missing_schema_pages)}\n"
        f"Recommendations: {len(result.top_recommendations)}",
        title="Audit Summary",
    ))

    console.print(f"\n[green]📄 Report:[/green] {paths['markdown']}")


@app.command(name="full-run")
def full_run(
    query: str = typer.Argument(help="Seed query for full GEO pipeline"),
    models: str = typer.Option("haiku", "--models", "-m", help="Comma-separated model aliases"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip ingestion if KB already populated"),
    skip_rate_limits: bool = typer.Option(
        True,
        "--skip-rate-limits/--fail-on-rate-limits",
        help="Continue and write a partial report when provider quotas/rate limits are hit",
    ),
):
    """🚀 Run the full GEO pipeline: expand → capture → analyze."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    console.print(Panel(
        f"[bold magenta]Full GEO Pipeline[/bold magenta]\n"
        f"Query: {query}\n"
        f"Models: {models}",
        title="🚀 Citewatch — Full Run",
    ))

    # Step 1: Expand
    console.print("\n[bold]━━━ Phase 1: Query Expansion ━━━[/bold]")
    from citewatch.expander import expand_query
    from citewatch.embeddings import get_embedding_manager

    model_aliases = [m.strip() for m in models.split(",")]
    resolved = cfg.models.resolve_model_ids(model_aliases)
    if not resolved:
        console.print(f"[red]❌ No valid models found for aliases: {models}[/red]")
        console.print(f"Available: {', '.join(m.alias for m in cfg.models.get_all())}")
        raise typer.Exit(1)

    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)
    try:
        emb.health_check()
    except RuntimeError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)

    variants = _run_async(
        expand_query(
            query,
            preferred_provider=_preferred_provider_for_models(resolved),
            preferred_model_id=_preferred_model_id_for_models(resolved),
        )
    )
    all_queries = [query] + [v.variant_text for v in variants]

    table = Table(title="Query Variants")
    table.add_column("Category", style="cyan")
    table.add_column("Query", style="white")
    for v in variants:
        table.add_row(v.category, v.variant_text)
    console.print(table)

    # Step 2: Capture
    console.print("\n[bold]━━━ Phase 2: LLM Capture ━━━[/bold]")

    from citewatch.cache import CacheManager, KnowledgeBase
    from citewatch.capture.anthropic import AnthropicCaptureEngine
    from citewatch.capture.google import GoogleCaptureEngine

    cache = CacheManager(db_path=cfg.cache.db_path, ttl_days=cfg.cache.ttl_days)
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

    all_responses = []
    capture_failures = []
    intended_models = [m.model_id for m in resolved]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for q in all_queries:
            for model in resolved:
                engine = engines.get(model.provider)
                if not engine:
                    continue
                task = progress.add_task(f"{model.alias}: {q[:35]}...", total=None)
                try:
                    resp = _run_async(engine.capture(query=q, model_id=model.model_id))
                    all_responses.append(resp)
                except Exception as e:
                    failure = _make_failure("capture", model.alias, q, e)
                    capture_failures.append(failure)
                    style = "yellow" if failure["error_type"] == "rate_limit" else "red"
                    prefix = "Skipping" if failure["error_type"] == "rate_limit" else "Failed"
                    console.print(f"[{style}]⚠️ {prefix} {model.alias}: {failure['message']}[/{style}]")
                progress.remove_task(task)

    console.print(f"[green]✅ Captured {len(all_responses)} responses[/green]")

    # Step 3: Analyze
    console.print("\n[bold]━━━ Phase 3: GEO Analysis ━━━[/bold]")
    kb = KnowledgeBase(db_path=cfg.ingestion.db_path)

    from citewatch.analyzer.reporter import generate_geo_report
    analysis_error = ""
    run_notes = []

    if not all_responses:
        analysis_error = "No model responses were captured, so GEO analysis was skipped."
        run_notes.append(analysis_error)
        analysis = _build_partial_analysis(
            query,
            all_responses,
            analysis_error,
            intended_models=intended_models,
        )
    else:
        from citewatch.analyzer.geo import analyze_query
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Running GEO judge...", total=None)
            try:
                analysis = _run_async(
                    analyze_query(
                        query,
                        all_responses,
                        kb,
                        preferred_provider=_preferred_provider_for_models(resolved),
                        preferred_model_id=_preferred_model_id_for_models(resolved),
                    )
                )
                if _analysis_parse_failed(analysis):
                    analysis_error = analysis.why_brand_missing
                    run_notes.append("The GEO judge response was truncated or unparsable.")
            except Exception as e:
                analysis_error = _compact_error(e)
                run_notes.append("The GEO judge failed before a full analysis could be produced.")
                if _is_rate_limit_error(e):
                    run_notes.append("A provider quota or rate limit was hit during analysis.")
                analysis = _build_partial_analysis(
                    query,
                    all_responses,
                    f"GEO judge did not complete: {analysis_error}",
                    intended_models=intended_models,
                )
                if not (skip_rate_limits and _is_rate_limit_error(e)):
                    paths = generate_geo_report(
                        query,
                        analysis,
                        all_responses,
                        output_dir=cfg.reports.output_dir,
                        report_status="partial",
                        run_notes=run_notes,
                        capture_failures=capture_failures,
                        analysis_error=analysis_error,
                        intended_models=intended_models,
                    )
                    console.print("[yellow]⚠️ Wrote a partial report before stopping[/yellow]")
                    console.print(f"  Markdown: {paths['markdown']}")
                    console.print(f"  JSON: {paths['json']}")
                    raise typer.Exit(1) from e

    paths = generate_geo_report(
        query,
        analysis,
        all_responses,
        output_dir=cfg.reports.output_dir,
        report_status="partial" if analysis_error else "complete",
        run_notes=run_notes,
        capture_failures=capture_failures,
        analysis_error=analysis_error,
        intended_models=intended_models,
    )

    # Final summary
    console.print(Panel(
        f"[bold]Status:[/] {'COMPLETE' if not analysis_error else 'PARTIAL'}\n"
        f"[bold]Citation Status:[/] {analysis.citation_status.value.upper()}\n"
        f"[bold]Brand Mentioned:[/] {'✅' if analysis.brand_mentioned else '❌'}\n"
        f"[bold]Confidence:[/] {analysis.confidence:.0%}\n"
        f"[bold]Responses:[/] {len(all_responses)}\n"
        f"[bold]Fixes:[/] {len(analysis.recommended_fixes)}",
        title="🎯 GEO Analysis Complete" if not analysis_error else "⚠️ Partial GEO Report",
    ))

    if analysis.recommended_fixes:
        table = Table(title="Top Recommendations", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Priority", style="bold", width=10)
        table.add_column("Action")
        for i, fix in enumerate(analysis.recommended_fixes[:5], 1):
            table.add_row(str(i), fix.priority.value.upper(), fix.action)
        console.print(table)

    if analysis_error:
        console.print(f"[yellow]⚠️ {analysis_error}[/yellow]")
    console.print(f"\n[green]📄 Full report: {paths['markdown']}[/green]")


@app.command()
def competitive(
    gsc_csv: Optional[str] = typer.Option(None, "--gsc-csv", help="Path to Google Search Console CSV export"),
    ahrefs_csv: Optional[str] = typer.Option(None, "--ahrefs-csv", help="Path to Ahrefs CSV export"),
    keywords: str = typer.Option("", "--keywords", "-k", help="Comma-separated list of keywords to analyze (e.g. 'best notes app,notion vs obsidian')"),
    competitors: str = typer.Option("", "--competitors", "-c", help="Comma-separated competitor domains (overrides config)"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Number of article comparisons to generate"),
    crawl_competitors: bool = typer.Option(False, "--crawl-competitors", help="Crawl competitor sites for content comparison"),
    models: str = typer.Option("flash", "--models", "-m", help="Model alias for LLM-powered analysis"),
    max_urls_per_domain: int = typer.Option(50, "--max-urls", help="Max competitor URLs to crawl per domain"),
    skip_rate_limits: bool = typer.Option(
        True,
        "--skip-rate-limits/--fail-on-rate-limits",
        help="Continue when provider rate limits are hit",
    ),
):
    """🏆 Run competitive SEO analysis: keyword gaps, competitor content comparison, and actionable recommendations."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    # Parse user-provided keywords
    user_keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()] if keywords else []

    console.print(Panel(
        f"[bold magenta]Competitive SEO Analysis[/bold magenta]\n"
        f"Keywords: {', '.join(user_keywords[:5]) + ('...' if len(user_keywords) > 5 else '') if user_keywords else '(from CSV/KB)'}\n"
        f"GSC CSV: {gsc_csv or '(none)'}\n"
        f"Ahrefs CSV: {ahrefs_csv or '(none)'}\n"
        f"Crawl competitors: {'Yes' if crawl_competitors else 'No'}\n"
        f"Top comparisons: {top_n}",
        title="🏆 Citewatch — Competitive Analysis",
    ))

    # Resolve models for LLM analysis
    model_aliases = [m.strip() for m in models.split(",")]
    resolved = cfg.models.resolve_model_ids(model_aliases)

    # Determine competitor domains
    comp_domains = [c.strip() for c in competitors.split(",") if c.strip()] if competitors else [
        c.domain for c in cfg.competitors
    ]
    if not comp_domains:
        console.print("[yellow]⚠️ No competitor domains configured. Add them to config.yaml or use --competitors[/yellow]")

    # Initialize embeddings
    from citewatch.embeddings import get_embedding_manager
    emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)
    try:
        emb.health_check()
    except RuntimeError as e:
        console.print(f"[red]❌ Embeddings not available: {e}[/red]")
        raise typer.Exit(1)

    # ── Phase 1: Import Keywords ─────────────────────────────────────
    console.print("\n[bold]━━━ Phase 1: Keyword Import ━━━[/bold]")

    from citewatch.audit.keyword_map import load_gsc_keywords, load_ahrefs_keywords

    our_keywords = []

    # User-provided keywords take first priority
    if user_keywords:
        from citewatch.models import KeywordEntry
        for kw_text in user_keywords:
            our_keywords.append(KeywordEntry(keyword=kw_text, url="", source="cli"))
        console.print(f"[green]✅ CLI keywords: {len(user_keywords)} loaded[/green]")

    if gsc_csv:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Loading GSC data...", total=None)
            gsc_kws = load_gsc_keywords(gsc_csv)
            our_keywords.extend(gsc_kws)
        console.print(f"[green]✅ GSC: {len(gsc_kws)} keywords loaded[/green]")

    if ahrefs_csv:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Loading Ahrefs data...", total=None)
            ahrefs_kws = load_ahrefs_keywords(ahrefs_csv)
            our_keywords.extend(ahrefs_kws)
        console.print(f"[green]✅ Ahrefs: {len(ahrefs_kws)} keywords loaded[/green]")

    if not our_keywords:
        console.print("[yellow]ℹ️  No CSV data provided — running with KB-derived keywords only[/yellow]")
        # Derive keywords from KB page titles
        from citewatch.cache import KnowledgeBase
        kb = KnowledgeBase(db_path=cfg.ingestion.db_path)
        if kb.count > 0:
            # Query KB for all unique page URLs/titles to build a basic keyword map
            from citewatch.models import KeywordEntry
            kb_results = kb.query(_kb_seed_query(cfg), n_results=50)
            if kb_results and kb_results.get("metadatas"):
                seen_urls = set()
                for meta in kb_results["metadatas"][0]:
                    url = meta.get("url", "")
                    heading = meta.get("heading", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        # Use heading as a pseudo-keyword
                        kw_text = heading if heading else url.split("/")[-1].replace("-", " ")
                        if kw_text:
                            our_keywords.append(KeywordEntry(keyword=kw_text, url=url, source="kb"))
            console.print(f"[cyan]📚 Derived {len(our_keywords)} keywords from KB[/cyan]")
        else:
            console.print("[yellow]⚠️ KB empty — run 'citewatch ingest' first for best results[/yellow]")

    # ── Phase 2: Keyword Clustering ──────────────────────────────────
    console.print("\n[bold]━━━ Phase 2: Keyword Clustering ━━━[/bold]")

    from citewatch.audit.keyword_map import cluster_keywords, detect_keyword_gaps
    from citewatch.models import CompetitiveAuditResult

    clusters = []
    if our_keywords:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task(f"Clustering {len(our_keywords)} keywords...", total=None)
            clusters = cluster_keywords(our_keywords)

        # Show cluster summary
        cannibalized = [c for c in clusters if c.is_cannibalized]
        console.print(f"[green]✅ {len(clusters)} keyword clusters found[/green]")
        if cannibalized:
            console.print(f"[yellow]⚠️ {len(cannibalized)} clusters have cannibalization issues[/yellow]")

        table = Table(title="Top Keyword Clusters", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Primary Keyword", style="cyan")
        table.add_column("Keywords", style="white", justify="right", width=8)
        table.add_column("Volume", style="yellow", justify="right", width=8)
        table.add_column("Pages", style="green", justify="right", width=6)
        table.add_column("Status", width=14)
        for i, c in enumerate(clusters[:10], 1):
            status = "[red]⚠️ Cannib.[/red]" if c.is_cannibalized else "[green]OK[/green]"
            table.add_row(str(i), c.primary_keyword, str(len(c.keywords)), str(c.total_volume), str(len(c.our_pages)), status)
        console.print(table)

    # ── Phase 3: Competitor Crawling ─────────────────────────────────
    competitor_pages = []
    if crawl_competitors and comp_domains:
        console.print("\n[bold]━━━ Phase 3: Competitor Crawling ━━━[/bold]")

        from citewatch.audit.competitor import discover_competitor_urls, crawl_competitor_content, CompetitorKB

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task(f"Discovering URLs from {len(comp_domains)} competitor sites...", total=None)
            domain_urls = _run_async(discover_competitor_urls(
                comp_domains,
                keyword_clusters=clusters or None,
                max_urls_per_domain=max_urls_per_domain,
            ))

        total_urls = sum(len(urls) for urls in domain_urls.values())
        console.print(f"[cyan]📝 Found {total_urls} relevant competitor URLs[/cyan]")

        for domain, urls in domain_urls.items():
            console.print(f"  • {domain}: {len(urls)} URLs")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task(f"Crawling {total_urls} competitor pages...", total=None)
            competitor_pages = _run_async(crawl_competitor_content(
                domain_urls,
                max_concurrent=3,
                request_delay=1.0,
            ))

        console.print(f"[green]✅ Crawled {len(competitor_pages)} competitor pages[/green]")

        # Store in competitor KB
        comp_kb = CompetitorKB(db_path=cfg.ingestion.competitor_db_path, embedding_mgr=emb)
        comp_kb.add_pages(competitor_pages)
        console.print(f"[cyan]📚 Competitor KB: {comp_kb.count} entries[/cyan]")
    elif not crawl_competitors and comp_domains:
        console.print("\n[dim]━━━ Phase 3: Competitor Crawling (skipped — use --crawl-competitors) ━━━[/dim]")

    # ── Phase 4: Keyword Gap Detection ───────────────────────────────
    keyword_gaps = []
    if our_keywords and competitor_pages:
        console.print("\n[bold]━━━ Phase 4: Keyword Gap Analysis ━━━[/bold]")

        # Build competitor keyword entries from crawled pages
        from citewatch.models import KeywordEntry
        comp_keywords = []
        for page in competitor_pages:
            if page.title:
                comp_keywords.append(KeywordEntry(
                    keyword=page.title,
                    url=page.url,
                    source="competitor",
                ))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Detecting keyword gaps...", total=None)
            keyword_gaps = detect_keyword_gaps(our_keywords, comp_keywords, our_domain=cfg.target.domain)

        console.print(f"[green]✅ {len(keyword_gaps)} keyword gaps detected[/green]")

        if keyword_gaps:
            table = Table(title="Top Keyword Gaps", show_lines=True)
            table.add_column("Keyword", style="cyan")
            table.add_column("Competitor", style="red")
            table.add_column("Their Pos", justify="right", width=9)
            table.add_column("Opportunity", style="yellow", justify="right", width=11)
            for gap in keyword_gaps[:8]:
                table.add_row(gap.keyword[:40], gap.competitor_domain, f"{gap.competitor_position:.0f}", f"{gap.opportunity_score:.2f}")
            console.print(table)

    # ── Phase 5: Article-Level Comparison ────────────────────────────
    article_comparisons = []
    if competitor_pages and resolved:
        console.print("\n[bold]━━━ Phase 5: Article Comparison (LLM-powered) ━━━[/bold]")

        from citewatch.audit.competitor import match_articles
        from citewatch.audit.competitive_analyzer import analyze_competitive_batch

        # Get our pages from KB for matching
        from citewatch.cache import KnowledgeBase
        kb = KnowledgeBase(db_path=cfg.ingestion.db_path)

        # Re-crawl our pages (or use KB-derived data)
        our_pages = []
        from citewatch.models import CrawledPage
        if kb.count > 0:
            kb_results = kb.query(_kb_seed_query(cfg), n_results=100)
            if kb_results and kb_results.get("metadatas"):
                seen = set()
                for meta, doc in zip(kb_results["metadatas"][0], kb_results["documents"][0]):
                    url = meta.get("url", "")
                    if url and url not in seen:
                        seen.add(url)
                        our_pages.append(CrawledPage(
                            url=url,
                            title=meta.get("heading", ""),
                            body_text=doc,
                            word_count=len(doc.split()),
                        ))

        if our_pages:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                progress.add_task("Matching articles...", total=None)
                matches = match_articles(our_pages, competitor_pages, clusters)

            console.print(f"[cyan]📝 {len(matches)} article pairs matched[/cyan]")

            if matches:
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                    progress.add_task(f"Analyzing top {min(top_n, len(matches))} pairs via LLM...", total=None)
                    try:
                        article_comparisons = _run_async(analyze_competitive_batch(
                            matches,
                            top_n=top_n,
                            preferred_provider=_preferred_provider_for_models(resolved),
                            preferred_model_id=_preferred_model_id_for_models(resolved),
                        ))
                    except Exception as e:
                        err_msg = _compact_error(e)
                        if skip_rate_limits and _is_rate_limit_error(e):
                            console.print(f"[yellow]⚠️ Rate limit hit during analysis: {err_msg}[/yellow]")
                        else:
                            console.print(f"[red]❌ Article analysis failed: {err_msg}[/red]")

                console.print(f"[green]✅ {len(article_comparisons)} comparisons generated[/green]")
        else:
            console.print("[yellow]⚠️ No KB pages available for matching — run 'citewatch ingest' first[/yellow]")

    # ── Phase 6: Generate Report ─────────────────────────────────────
    console.print("\n[bold]━━━ Phase 6: Report Generation ━━━[/bold]")

    from citewatch.audit.competitive_analyzer import generate_competitive_recommendations
    from citewatch.analyzer.reporter import generate_competitive_report

    # Build cannibalization clusters from keyword clusters
    from citewatch.models import CannibalizationCluster
    cannib_clusters = []
    for c in clusters:
        if c.is_cannibalized:
            cannib_clusters.append(CannibalizationCluster(
                queries=c.keywords[:5],
                pages=c.our_pages,
                max_similarity=0.85,
                recommendation=f"Consolidate {len(c.our_pages)} pages targeting '{c.primary_keyword}' into one authoritative page",
            ))

    # Generate prioritized recommendations
    recommendations = generate_competitive_recommendations(
        article_comparisons, keyword_gaps, cannib_clusters,
    )

    result = CompetitiveAuditResult(
        keyword_map=our_keywords,
        keyword_gaps=keyword_gaps,
        keyword_clusters=clusters,
        competitor_articles=[],  # Don't store full body text in report JSON
        article_comparisons=article_comparisons,
        cannibalization_clusters=cannib_clusters,
        top_recommendations=recommendations,
        domains_analyzed=comp_domains,
        total_keywords_analyzed=len(our_keywords),
    )

    paths = generate_competitive_report(result, output_dir=cfg.reports.output_dir)

    # ── Summary ──────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold green]✅ Competitive Analysis Complete[/bold green]\n"
        f"Keywords analyzed: {len(our_keywords)}\n"
        f"Keyword clusters: {len(clusters)}\n"
        f"Keyword gaps: {len(keyword_gaps)}\n"
        f"Cannibalization issues: {len(cannib_clusters)}\n"
        f"Competitor pages crawled: {len(competitor_pages)}\n"
        f"Article comparisons: {len(article_comparisons)}\n"
        f"Priority recommendations: {len(recommendations)}",
        title="🏆 Competitive Analysis Summary",
    ))

    if recommendations:
        table = Table(title="Top Recommendations", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Priority", style="bold", width=10)
        table.add_column("Action")
        for i, fix in enumerate(recommendations[:8], 1):
            table.add_row(str(i), fix.priority.value.upper(), fix.action)
        console.print(table)

    console.print(f"\n[green]📄 Full report: {paths['markdown']}[/green]")
    console.print(f"[green]📄 JSON data: {paths['json']}[/green]")


@app.command()
def status():
    """📊 Show current status of cache, KB, and configuration."""
    from citewatch.config import settings as get_cfg
    cfg = get_cfg()

    console.print(Panel("[bold cyan]Citewatch Status[/]", title="Status"))

    # Config
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Target", cfg.target.domain)
    table.add_row("Anthropic Key", "✅ Set" if cfg.anthropic_api_key else "❌ Not set")
    table.add_row("Google Key", "✅ Set" if cfg.google_api_key else "❌ Not set")
    table.add_row("Ollama URL", cfg.ollama_base_url)
    table.add_row("Models", ", ".join(m.alias for m in cfg.models.get_all()))
    table.add_row("Competitors", ", ".join(cfg.get_competitor_names()))
    console.print(table)

    # Check Ollama
    try:
        from citewatch.embeddings import get_embedding_manager
        emb = get_embedding_manager(ollama_url=cfg.ollama_base_url)
        emb.health_check()
        console.print("[green]✅ Ollama: healthy[/green]")
    except Exception:
        console.print("[red]❌ Ollama: not reachable[/red]")

    # Cache status
    try:
        from citewatch.cache import CacheManager, KnowledgeBase
        cache = CacheManager(db_path=cfg.cache.db_path, ttl_days=cfg.cache.ttl_days)
        console.print(f"[cyan]📦 Cache: {cache.count} entries[/cyan]")
    except Exception:
        console.print("[yellow]⚠️ Cache: not initialized[/yellow]")

    # KB status
    try:
        kb = KnowledgeBase(db_path=cfg.ingestion.db_path)
        console.print(f"[cyan]📚 Knowledge Base: {kb.count} chunks[/cyan]")
    except Exception:
        console.print("[yellow]⚠️ KB: not initialized[/yellow]")

    # Competitor KB status
    try:
        from citewatch.audit.competitor import CompetitorKB
        comp_kb = CompetitorKB(db_path=cfg.ingestion.competitor_db_path)
        console.print(f"[cyan]🏆 Competitor KB: {comp_kb.count} entries[/cyan]")
    except Exception:
        console.print("[dim]🏆 Competitor KB: not initialized[/dim]")


if __name__ == "__main__":
    app()

