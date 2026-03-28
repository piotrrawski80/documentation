"""
scripts/parse_docs.py — Parse local RHEL HTML documentation into structured JSON.

This script focuses ONLY on parsing — no embeddings, no database, no search.
It reads locally cached HTML files, extracts structured sections, and saves
the output as JSON files ready for the chunking pipeline.

Usage:
    # Parse all RHEL 9 guides from local docs
    python scripts/parse_docs.py --docs-dir D:/cloude/dokumentacja/rhel-docs-rhel9/raw --version 9

    # Parse only first 3 guides (for testing)
    python scripts/parse_docs.py --docs-dir D:/cloude/dokumentacja/rhel-docs-rhel9/raw --version 9 --limit 3

    # Parse a single guide
    python scripts/parse_docs.py --docs-dir D:/cloude/dokumentacja/rhel-docs-rhel9/raw --version 9 --guide configuring_and_managing_networking

    # Custom output directory
    python scripts/parse_docs.py --docs-dir D:/cloude/dokumentacja/rhel-docs-rhel9/raw --version 9 --output data/parsed

    # Show stats without saving
    python scripts/parse_docs.py --docs-dir D:/cloude/dokumentacja/rhel-docs-rhel9/raw --version 9 --stats-only
"""

import sys
import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rh_linux_docs_agent.parser.html_parser import parse_guide_html
from rh_linux_docs_agent.parser.models import ParsedGuide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="parse-docs",
    help="Parse local RHEL HTML documentation into structured JSON.",
    add_completion=False,
)
console = Console()


def discover_local_guides(docs_dir: Path, version: str) -> list[tuple[str, Path]]:
    """
    Find all guide HTML files in the local documentation directory.

    Expected structure:
      docs_dir/docs.redhat.com/en/documentation/red_hat_enterprise_linux/{version}/html-single/{slug}/index.html

    Returns:
        List of (slug, html_path) tuples sorted by slug name.
    """
    html_single_dir = (
        docs_dir / "docs.redhat.com" / "en" / "documentation"
        / "red_hat_enterprise_linux" / version / "html-single"
    )

    if not html_single_dir.exists():
        console.print(
            f"[red]Error:[/red] Directory not found: {html_single_dir}\n"
            f"Expected RHEL {version} docs at this path."
        )
        raise typer.Exit(1)

    guides = []
    for entry in sorted(html_single_dir.iterdir()):
        if not entry.is_dir():
            continue
        index_html = entry / "index.html"
        if index_html.exists():
            guides.append((entry.name, index_html))

    return guides


@app.command()
def main(
    docs_dir: Path = typer.Option(
        ...,
        "--docs-dir",
        "-d",
        help="Root directory of scraped RHEL docs (contains docs.redhat.com/...).",
    ),
    version: str = typer.Option(
        "9",
        "--version",
        "-v",
        help="RHEL version to parse.",
    ),
    output: Path = typer.Option(
        Path("data/parsed"),
        "--output",
        "-o",
        help="Output directory for JSON files.",
    ),
    guide: str = typer.Option(
        "",
        "--guide",
        "-g",
        help="Parse only this specific guide slug.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        "-l",
        help="Only parse first N guides (for testing).",
    ),
    stats_only: bool = typer.Option(
        False,
        "--stats-only",
        help="Show parsing stats without saving JSON files.",
    ),
) -> None:
    """Parse local RHEL HTML documentation into structured JSON."""

    start_time = time.time()

    # Discover guides
    console.print(f"\n[bold]Discovering RHEL {version} guides in {docs_dir}...[/bold]")
    all_guides = discover_local_guides(docs_dir, version)
    console.print(f"  Found [bold]{len(all_guides)}[/bold] guides")

    # Filter by specific guide if requested
    if guide:
        all_guides = [(s, p) for s, p in all_guides if s == guide]
        if not all_guides:
            console.print(f"[red]Error:[/red] Guide '{guide}' not found.")
            raise typer.Exit(1)

    # Apply limit
    if limit > 0:
        all_guides = all_guides[:limit]
        console.print(f"  [dim]Limited to first {limit} guides[/dim]")

    # Parse each guide
    console.print(f"\n[bold]Parsing {len(all_guides)} guides...[/bold]")

    parsed_guides: list[ParsedGuide] = []
    errors: list[tuple[str, str]] = []

    for i, (slug, html_path) in enumerate(all_guides):
        try:
            parsed = parse_guide_html(
                html_path=html_path,
                slug=slug,
                version=version,
            )
            parsed_guides.append(parsed)

            # Progress output every guide
            console.print(
                f"  [{i + 1}/{len(all_guides)}] {slug}: "
                f"[bold]{parsed.total_sections}[/bold] sections, "
                f"[bold]{sum(len(s.code_blocks) for s in parsed.sections)}[/bold] code blocks"
            )

        except Exception as e:
            errors.append((slug, str(e)))
            console.print(f"  [{i + 1}/{len(all_guides)}] [red]ERROR[/red] {slug}: {e}")

    # Stats table
    _show_stats(parsed_guides, errors)

    # Save JSON output
    if not stats_only and parsed_guides:
        output_dir = output / f"rhel{version}"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold]Saving JSON to {output_dir}/...[/bold]")

        for pg in parsed_guides:
            out_file = output_dir / f"{pg.slug}.json"
            pg.save_json(out_file)

        # Also save a manifest
        from rh_linux_docs_agent.parser.models import PARSER_VERSION, now_iso
        manifest = {
            "product": "rhel",
            "major_version": version,
            "parser_version": PARSER_VERSION,
            "generated_at": now_iso(),
            "total_guides": len(parsed_guides),
            "total_sections": sum(g.total_sections for g in parsed_guides),
            "guides": [
                {
                    "slug": g.slug,
                    "title": g.title,
                    "doc_type": g.doc_type,
                    "minor_version": g.minor_version,
                    "sections": g.total_sections,
                    "guide_url": g.guide_url,
                }
                for g in parsed_guides
            ],
        }
        manifest_path = output_dir / "_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"  Saved {len(parsed_guides)} JSON files + manifest")

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")


def _show_stats(guides: list[ParsedGuide], errors: list[tuple[str, str]]) -> None:
    """Display full parsing statistics including record-level analysis."""
    if not guides:
        console.print("\n[yellow]No guides were parsed successfully.[/yellow]")
        return

    # ── Collect all flat records for analysis ─────────────────────────────
    all_records: list[dict] = []
    for g in guides:
        all_records.extend(g.section_records())

    total_sections = len(all_records)
    with_code = sum(1 for r in all_records if r["has_code_blocks"])
    with_tables = sum(1 for r in all_records if r["has_tables"])
    empty = sum(1 for r in all_records if r["char_count"] == 0)
    total_chars = sum(r["char_count"] for r in all_records)
    total_words = sum(r["word_count"] for r in all_records)
    total_code_blocks = sum(len(r["code_blocks"]) for r in all_records)

    # Content type breakdown
    type_counts: dict[str, int] = {}
    for r in all_records:
        ct = r["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    # doc_type breakdown
    doc_type_counts: dict[str, int] = {}
    for g in guides:
        doc_type_counts[g.doc_type] = doc_type_counts.get(g.doc_type, 0) + 1

    # ── Main summary table ────────────────────────────────────────────────
    console.print(f"\n[bold]Parse Summary:[/bold]")

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")

    t.add_row("Guides parsed", f"{len(guides)}")
    t.add_row("Parse errors", f"{len(errors)}")
    t.add_row("Section records", f"{total_sections:,}")
    t.add_row("Records with code_blocks", f"{with_code:,}")
    t.add_row("Records with tables", f"{with_tables:,}")
    t.add_row("Empty records (0 chars)", f"{empty}")
    t.add_row("Total code_blocks", f"{total_code_blocks:,}")
    t.add_row("Total chars", f"{total_chars:,}")
    t.add_row("Total words", f"{total_words:,}")
    t.add_row("Avg chars/section", f"{total_chars // max(total_sections, 1):,}")
    t.add_row("", "")
    for ct in ("procedure", "concept", "reference"):
        t.add_row(f"  {ct}", f"{type_counts.get(ct, 0):,}")

    console.print(t)

    # ── doc_type breakdown ────────────────────────────────────────────────
    console.print(f"\n[bold]Guides by doc_type:[/bold]")
    dt = Table(show_header=True, header_style="bold cyan")
    dt.add_column("doc_type", style="cyan")
    dt.add_column("guides", justify="right")
    for dtype, count in sorted(doc_type_counts.items(), key=lambda x: -x[1]):
        dt.add_row(dtype, str(count))
    console.print(dt)

    # ── Top 10 largest sections ───────────────────────────────────────────
    sorted_by_size = sorted(all_records, key=lambda r: r["char_count"], reverse=True)
    console.print(f"\n[bold]Top 10 largest sections by char_count:[/bold]")
    lt = Table(show_header=True, header_style="bold cyan")
    lt.add_column("#", style="dim", width=4)
    lt.add_column("chars", justify="right", width=9)
    lt.add_column("words", justify="right", width=9)
    lt.add_column("guide_slug", width=35, no_wrap=True, overflow="ellipsis")
    lt.add_column("heading", no_wrap=True, overflow="ellipsis")
    for i, r in enumerate(sorted_by_size[:10]):
        lt.add_row(
            str(i + 1),
            f"{r['char_count']:,}",
            f"{r['word_count']:,}",
            r["guide_slug"][:30],
            r["heading"][:50],
        )
    console.print(lt)

    # ── Errors ────────────────────────────────────────────────────────────
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        for slug, err in errors:
            console.print(f"  {slug}: {err}")


if __name__ == "__main__":
    app()
