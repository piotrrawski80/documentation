"""
scripts/scrape.py — Download and cache RHEL documentation HTML pages.

This script downloads the HTML documentation from docs.redhat.com and saves
it to the local cache directory. It ONLY downloads — it does not parse,
chunk, or embed anything.

Use this script when you want to:
- Pre-populate the cache for air-gapped deployment
- Update the cache after RHEL docs have been updated
- Download docs separately before running the full ingestion pipeline

Usage:
    # Download RHEL 9 docs
    python scripts/scrape.py --version 9

    # Download all versions
    python scripts/scrape.py --all-versions

    # Force re-download even if cached
    python scripts/scrape.py --version 10 --force-refresh

    # List what's in the cache
    python scripts/scrape.py --list-cache
"""

import sys
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add the src directory to the Python path so we can import our package.
# This is needed when running scripts directly (not via `python -m`).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rh_linux_docs_agent.scraper.discovery import discover_guides
from rh_linux_docs_agent.scraper.fetcher import fetch_all_guides, list_cached_guides
from rh_linux_docs_agent.config import settings

# Set up logging — shows timestamped messages in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="scrape",
    help="Download and cache RHEL documentation HTML pages.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    version: list[str] = typer.Option(
        [],
        "--version",
        "-v",
        help='RHEL version to scrape: "8", "9", or "10". Repeat for multiple versions.',
    ),
    all_versions: bool = typer.Option(
        False,
        "--all-versions",
        help="Scrape all supported RHEL versions (8, 9, 10).",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Re-download even if the HTML is already in the cache.",
    ),
    list_cache: bool = typer.Option(
        False,
        "--list-cache",
        help="Show what's currently in the HTML cache.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Only download the first N guides (useful for testing).",
    ),
    cache_dir: str = typer.Option(
        "",
        "--cache-dir",
        help="Override the cache directory (default from config).",
    ),
) -> None:
    """
    Download and cache RHEL documentation HTML pages.

    Downloads the single-page HTML version of each guide from docs.redhat.com
    and saves it locally. Already-cached guides are skipped unless --force-refresh.
    """

    # Override cache directory if specified
    if cache_dir:
        settings.cache_dir = Path(cache_dir)

    # ── List cache mode ────────────────────────────────────────────────────
    if list_cache:
        _show_cache_contents()
        return

    # ── Determine which versions to scrape ────────────────────────────────
    if all_versions:
        versions_to_scrape = settings.supported_versions
    elif version:
        versions_to_scrape = list(version)
    else:
        console.print(
            "[red]Error:[/red] Specify at least one --version or use --all-versions\n"
            "Example: python scripts/scrape.py --version 9"
        )
        raise typer.Exit(1)

    # Validate version numbers
    for v in versions_to_scrape:
        if v not in settings.supported_versions:
            console.print(
                f"[red]Error:[/red] Unsupported version '{v}'. "
                f"Supported: {settings.supported_versions}"
            )
            raise typer.Exit(1)

    # ── Scrape each version ────────────────────────────────────────────────
    total_downloaded = 0
    total_cached = 0

    for v in versions_to_scrape:
        console.print(f"\n[bold cyan]━━━ Scraping RHEL {v} ━━━[/bold cyan]")

        # Step 1: Discover all guides
        console.print(f"[Step 1] Discovering guides for RHEL {v}...")
        guides = discover_guides(v)

        if not guides:
            console.print(f"[yellow]Warning:[/yellow] No guides found for RHEL {v}. Skipping.")
            continue

        console.print(f"  Found [bold]{len(guides)}[/bold] guides")

        # Apply limit if specified
        if limit > 0:
            guides = guides[:limit]
            console.print(f"  [dim]Limited to first {limit} guides[/dim]")

        # Step 2: Download HTML pages
        console.print(f"\n[Step 2] Downloading HTML pages...")
        results = fetch_all_guides(guides, force_refresh=force_refresh)

        downloaded = sum(1 for g in guides if g.slug in results)
        cached = sum(1 for g in guides if g.slug not in results or not force_refresh)

        total_downloaded += downloaded
        console.print(
            f"  Downloaded: [bold green]{downloaded}[/bold green] | "
            f"Cached: [bold blue]{len(results) - downloaded if not force_refresh else 0}[/bold blue] | "
            f"Failed: [bold red]{len(guides) - len(results)}[/bold red]"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]✓ Scraping complete![/bold green]")
    console.print(f"  Cache directory: {settings.cache_dir.resolve()}")
    _show_cache_contents()


def _show_cache_contents() -> None:
    """Display a table of cached guides by version."""
    console.print("\n[bold]Cache Contents:[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Version", style="cyan", width=10)
    table.add_column("Cached Guides", justify="right")
    table.add_column("Cache Directory")

    total = 0
    for v in settings.supported_versions:
        cached = list_cached_guides(v)
        cache_path = settings.cache_dir / v
        table.add_row(
            f"RHEL {v}",
            str(len(cached)),
            str(cache_path),
        )
        total += len(cached)

    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "")
    console.print(table)


if __name__ == "__main__":
    app()
