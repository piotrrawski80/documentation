"""
scripts/ingest.py — Full ingestion pipeline: scrape → parse → chunk → embed → store.

This is the main script to run when setting up the system for the first time
or updating the documentation index. It runs all pipeline stages in order.

Pipeline stages:
  1. Discover guides (fetch list from docs.redhat.com)
  2. Download HTML pages (skip cached)
  3. Parse HTML → extract sections
  4. Chunk sections (150–800 tokens)
  5. Embed chunks (convert to 384-dim vectors)
  6. Store in LanceDB (upsert — safe to re-run)

Usage:
    # First-time setup: ingest RHEL 9
    python scripts/ingest.py --version 9

    # Test with a small subset first (faster, good for checking things work)
    python scripts/ingest.py --version 9 --limit 5

    # Ingest all versions
    python scripts/ingest.py --all-versions

    # Re-scrape (re-download HTML even if cached)
    python scripts/ingest.py --version 10 --force-refresh

    # See what's in the database
    python scripts/ingest.py --list

    # Delete a version and re-ingest
    python scripts/ingest.py --delete 9
    python scripts/ingest.py --version 9

    # Start completely fresh
    python scripts/ingest.py --fresh --version 9
"""

import sys
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rh_linux_docs_agent.scraper.discovery import discover_guides
from rh_linux_docs_agent.scraper.fetcher import fetch_all_guides, get_cached_html
from rh_linux_docs_agent.parser.html_parser import parse_guide_html
from rh_linux_docs_agent.chunker.splitter import chunk_guide
from rh_linux_docs_agent.indexer.embedder import Embedder
from rh_linux_docs_agent.indexer.store import DocStore
from rh_linux_docs_agent.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="ingest",
    help="Run the full RHEL docs ingestion pipeline.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    version: list[str] = typer.Option(
        [],
        "--version",
        "-v",
        help='RHEL version to ingest: "8", "9", or "10". Repeat for multiple.',
    ),
    all_versions: bool = typer.Option(
        False,
        "--all-versions",
        help="Ingest all supported versions (8, 9, 10).",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Re-download HTML even if already cached.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Only process first N guides (for testing).",
    ),
    list_db: bool = typer.Option(
        False,
        "--list",
        help="Show what's currently indexed in the database.",
    ),
    delete: str = typer.Option(
        "",
        "--delete",
        help="Delete all chunks for a version, e.g. --delete 9",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Drop the entire database and start fresh before ingesting.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Skip scraping; use only already-cached HTML files.",
    ),
    build_index: bool = typer.Option(
        True,
        "--build-index/--no-build-index",
        help="Build search indexes after ingestion (recommended).",
    ),
) -> None:
    """
    Run the full RHEL documentation ingestion pipeline.

    Downloads, parses, chunks, embeds, and stores RHEL documentation
    in a local vector database for fast retrieval.
    """

    store = DocStore()

    # ── List mode ──────────────────────────────────────────────────────────
    if list_db:
        _show_database_contents(store)
        return

    # ── Delete mode ────────────────────────────────────────────────────────
    if delete:
        if delete not in settings.supported_versions:
            console.print(
                f"[red]Error:[/red] Unknown version '{delete}'. "
                f"Supported: {settings.supported_versions}"
            )
            raise typer.Exit(1)
        console.print(f"[yellow]Deleting all RHEL {delete} chunks from database...[/yellow]")
        n = store.delete_by_version(delete)
        console.print(f"[green]✓[/green] Deleted chunks for RHEL {delete}")
        _show_database_contents(store)
        return

    # ── Fresh mode ─────────────────────────────────────────────────────────
    if fresh:
        console.print(
            Panel(
                "[bold red]WARNING: This will delete ALL data in the database.[/bold red]\n"
                "All indexed documentation will be removed.",
                title="Fresh Mode",
            )
        )
        confirm = typer.confirm("Are you sure you want to drop the entire database?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit(0)
        store.drop_table()
        console.print("[green]✓[/green] Database cleared")

    # ── Determine which versions to ingest ────────────────────────────────
    if all_versions:
        versions_to_ingest = settings.supported_versions
    elif version:
        versions_to_ingest = list(version)
    else:
        console.print(
            "[red]Error:[/red] Specify at least one --version or use --all-versions\n"
            "Example: python scripts/ingest.py --version 9\n"
            "Example: python scripts/ingest.py --list"
        )
        raise typer.Exit(1)

    for v in versions_to_ingest:
        if v not in settings.supported_versions:
            console.print(
                f"[red]Error:[/red] Unsupported version '{v}'. "
                f"Supported: {settings.supported_versions}"
            )
            raise typer.Exit(1)

    # ── Load embedding model (done once, shared across versions) ──────────
    console.print("\n[bold]Loading embedding model...[/bold]")
    embedder = Embedder()
    console.print(f"  [green]✓[/green] Model loaded: {settings.embedding_model}")

    # ── Process each version ───────────────────────────────────────────────
    grand_total_chunks = 0

    for v in versions_to_ingest:
        start_time = time.time()
        console.print(f"\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print(f"[bold cyan]  Ingesting RHEL {v}[/bold cyan]")
        console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        total_chunks = _ingest_version(
            version=v,
            embedder=embedder,
            store=store,
            force_refresh=force_refresh,
            limit=limit,
            offline=offline,
        )
        grand_total_chunks += total_chunks

        elapsed = time.time() - start_time
        console.print(
            f"\n[green]✓[/green] RHEL {v} complete: "
            f"[bold]{total_chunks:,}[/bold] chunks in {elapsed:.0f}s"
        )

    # ── Build search indexes ───────────────────────────────────────────────
    if build_index and grand_total_chunks > 0:
        console.print(f"\n[bold]Building search indexes...[/bold]")
        store.create_indexes()
        console.print("[green]✓[/green] Indexes built")

    # ── Final summary ─────────────────────────────────────────────────────
    console.print(f"\n[bold green]━━━ Ingestion Complete ━━━[/bold green]")
    console.print(f"  Total new chunks: [bold]{grand_total_chunks:,}[/bold]")
    _show_database_contents(store)
    console.print(
        "\n[dim]Start the agent with: python -m rh_linux_docs_agent.agent.app[/dim]"
    )


def _ingest_version(
    version: str,
    embedder: Embedder,
    store: DocStore,
    force_refresh: bool,
    limit: int,
    offline: bool,
) -> int:
    """
    Run the full ingestion pipeline for a single RHEL version.

    Args:
        version: RHEL major version string.
        embedder: Loaded Embedder instance.
        store: DocStore instance for database operations.
        force_refresh: Re-download cached HTML if True.
        limit: Max guides to process (0 = no limit).
        offline: Skip discovery/download, use only cached HTML.

    Returns:
        Total number of chunks inserted for this version.
    """

    # ── Step 1: Discover guides ────────────────────────────────────────────
    if offline:
        console.print(f"\n[Step 1/6] Loading guides from cache (offline mode)...")
        from rh_linux_docs_agent.scraper.fetcher import list_cached_guides
        from rh_linux_docs_agent.scraper.discovery import GuideInfo

        cached_slugs = list_cached_guides(version)
        if not cached_slugs:
            console.print(
                f"[yellow]Warning:[/yellow] No cached HTML found for RHEL {version} "
                f"in {settings.cache_dir / version}. "
                "Run without --offline first to download the docs."
            )
            return 0

        guides = [
            GuideInfo(
                slug=slug,
                title=slug.replace("_", " ").title(),
                version=version,
                url=f"{settings.docs_base_url}/{version}/html-single/{slug}/index",
            )
            for slug in cached_slugs
        ]
        console.print(f"  Found [bold]{len(guides)}[/bold] cached guides")
    else:
        console.print(f"\n[Step 1/6] Discovering guides for RHEL {version}...")
        guides = discover_guides(version)

        if not guides:
            console.print(
                f"[red]Error:[/red] No guides found for RHEL {version}. "
                "Check your internet connection and try again."
            )
            return 0
        console.print(f"  Found [bold]{len(guides)}[/bold] guides")

    # Apply limit for testing
    if limit > 0:
        guides = guides[:limit]
        console.print(f"  [dim]Limited to first {limit} guides[/dim]")

    # ── Step 2: Download HTML ──────────────────────────────────────────────
    if offline:
        console.print(f"\n[Step 2/6] Using cached HTML (offline mode — skipping download)")
    else:
        console.print(f"\n[Step 2/6] Downloading HTML pages...")
        fetch_results = fetch_all_guides(guides, force_refresh=force_refresh)
        downloaded = sum(1 for g in guides if g.slug in fetch_results)
        console.print(
            f"  Downloaded: {downloaded} | "
            f"Already cached: {len(fetch_results) - downloaded if not force_refresh else 0} | "
            f"Failed: {len(guides) - len(fetch_results)}"
        )

    # ── Step 3: Parse HTML ─────────────────────────────────────────────────
    console.print(f"\n[Step 3/6] Parsing HTML...")
    all_guides_parsed = []

    for i, guide in enumerate(guides):
        html = get_cached_html(version, guide.slug)
        if html is None:
            logger.warning(f"No cached HTML for {guide.slug} — skipping")
            continue

        try:
            parsed = parse_guide_html(
                html=html,
                slug=guide.slug,
                version=version,
                url=guide.url,
                guide_title=guide.title,
            )
            all_guides_parsed.append(parsed)

            if (i + 1) % 10 == 0 or i == len(guides) - 1:
                console.print(
                    f"  [{i + 1}/{len(guides)}] Parsed {len(all_guides_parsed)} guides so far"
                )
        except Exception as e:
            logger.warning(f"Parse error for {guide.slug}: {e}")

    total_sections = sum(g.total_sections for g in all_guides_parsed)
    console.print(
        f"  Parsed [bold]{len(all_guides_parsed)}[/bold] guides → "
        f"[bold]{total_sections:,}[/bold] sections"
    )

    # ── Step 4: Chunk sections ─────────────────────────────────────────────
    console.print(f"\n[Step 4/6] Chunking sections...")
    all_chunks = []

    for parsed_guide in all_guides_parsed:
        chunks = chunk_guide(parsed_guide)
        all_chunks.extend(chunks)

    console.print(
        f"  Created [bold]{len(all_chunks):,}[/bold] chunks from "
        f"{len(all_guides_parsed)} guides"
    )

    if not all_chunks:
        console.print("[yellow]Warning:[/yellow] No chunks created. Nothing to index.")
        return 0

    # ── Step 5: Embed chunks ───────────────────────────────────────────────
    console.print(f"\n[Step 5/6] Embedding chunks...")
    texts = [chunk.text for chunk in all_chunks]
    vectors = embedder.embed_with_progress(texts, description=f"Embedding RHEL {version}")

    console.print(f"  Embedded [bold]{len(vectors):,}[/bold] chunks")

    # ── Step 6: Store in LanceDB ───────────────────────────────────────────
    console.print(f"\n[Step 6/6] Storing in LanceDB...")
    n_inserted = store.insert_chunks(all_chunks, vectors)

    total_in_db = store.get_total_count()
    console.print(
        f"  Inserted [bold]{n_inserted:,}[/bold] records\n"
        f"  Total records in table: [bold]{total_in_db:,}[/bold]"
    )

    return n_inserted


def _show_database_contents(store: DocStore) -> None:
    """Display a table showing what's currently in the database."""
    console.print("\n[bold]Database Contents:[/bold]")

    versions = store.list_versions()

    if not versions:
        console.print("  [dim]Database is empty. Run ingestion to populate it.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Version", style="cyan", width=12)
    table.add_column("Chunks", justify="right")

    total = 0
    for v, count in sorted(versions.items()):
        table.add_row(f"RHEL {v}", f"{count:,}")
        total += count

    table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]")
    console.print(table)


if __name__ == "__main__":
    app()
