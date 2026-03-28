"""
scripts/search_cli.py — Interactive command-line search tool.

Lets you search the RHEL documentation directly from the terminal
without starting the full web UI. Good for quick lookups and testing
search quality.

Usage:
    # Interactive mode (type queries, press Enter, Ctrl+C to exit)
    python scripts/search_cli.py

    # Single query
    python scripts/search_cli.py --query "configure static IP"

    # Filter by version
    python scripts/search_cli.py --query "firewalld zones" --version 9

    # Compare across versions
    python scripts/search_cli.py --compare "dnf vs yum" --versions 8,9

    # Show more results
    python scripts/search_cli.py --query "SELinux" --top-n 10
"""

import sys
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rh_linux_docs_agent.search.hybrid import HybridSearch
from rh_linux_docs_agent.config import settings

logging.basicConfig(level=logging.WARNING)  # Suppress info logs in interactive mode

app = typer.Typer(
    name="search",
    help="Search RHEL documentation from the command line.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    query: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Search query. If omitted, runs in interactive mode.",
    ),
    version: str = typer.Option(
        "",
        "--version",
        "-v",
        help='Filter by RHEL version: "8", "9", or "10".',
    ),
    compare: str = typer.Option(
        "",
        "--compare",
        help="Compare a topic across versions.",
    ),
    versions: str = typer.Option(
        "",
        "--versions",
        help='Versions to compare (comma-separated, e.g. "8,9,10").',
    ),
    top_n: int = typer.Option(
        5,
        "--top-n",
        "-n",
        help="Number of results to show.",
    ),
) -> None:
    """
    Search RHEL documentation from the command line.
    """
    # Initialize search (loads the embedding model)
    console.print("[dim]Loading search engine...[/dim]", end="\r")
    try:
        searcher = HybridSearch()
    except Exception as e:
        console.print(f"[red]Error initializing search:[/red] {e}")
        console.print("Make sure you've run: python scripts/ingest.py --version 9")
        raise typer.Exit(1)
    console.print("[green]✓[/green] Search engine ready           ")

    # ── Compare mode ───────────────────────────────────────────────────────
    if compare:
        compare_versions = [v.strip() for v in versions.split(",")] if versions else settings.supported_versions
        _run_compare(searcher, compare, compare_versions, top_n)
        return

    # ── Single query mode ──────────────────────────────────────────────────
    if query:
        _run_search(searcher, query, version or None, top_n)
        return

    # ── Interactive mode ───────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold]RHEL Documentation Search[/bold]\n"
            "Type a query and press Enter. Commands:\n"
            "  [cyan]:v 9[/cyan]  — filter by version (e.g., :v 9)\n"
            "  [cyan]:v all[/cyan] — search all versions\n"
            "  [cyan]:n 10[/cyan] — show 10 results\n"
            "  [cyan]:q[/cyan]    — quit",
            title="Interactive Search",
        )
    )

    current_version: str | None = None
    current_top_n = top_n

    while True:
        try:
            user_input = console.input("\n[bold cyan]Search:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith(":"):
            parts = user_input[1:].split()
            cmd = parts[0].lower() if parts else ""

            if cmd == "q" or cmd == "quit":
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "v" and len(parts) > 1:
                ver = parts[1]
                if ver == "all":
                    current_version = None
                    console.print("[dim]Version filter: all versions[/dim]")
                elif ver in settings.supported_versions:
                    current_version = ver
                    console.print(f"[dim]Version filter: RHEL {current_version}[/dim]")
                else:
                    console.print(f"[red]Unknown version '{ver}'. Options: {settings.supported_versions}[/red]")
            elif cmd == "n" and len(parts) > 1:
                try:
                    current_top_n = int(parts[1])
                    console.print(f"[dim]Showing {current_top_n} results[/dim]")
                except ValueError:
                    console.print(f"[red]Invalid number: {parts[1]}[/red]")
            else:
                console.print("[red]Unknown command. Type :q to quit.[/red]")
            continue

        _run_search(searcher, user_input, current_version, current_top_n)


def _run_search(
    searcher: HybridSearch,
    query: str,
    version: str | None,
    top_n: int,
) -> None:
    """Run a search and print formatted results."""
    version_label = f"RHEL {version}" if version else "all versions"
    console.print(f"\n[dim]Searching {version_label} for: {query}[/dim]")

    results = searcher.search(query, version=version, top_n=top_n)

    if not results:
        console.print(
            f"[yellow]No results found.[/yellow] "
            "Check that the database is populated: python scripts/ingest.py --list"
        )
        return

    console.print(Rule(f"[bold]Top {len(results)} Results[/bold]"))

    for i, r in enumerate(results, start=1):
        hierarchy = r.get("section_hierarchy", [])
        if isinstance(hierarchy, str):
            import json
            try:
                hierarchy = json.loads(hierarchy)
            except Exception:
                hierarchy = []

        section_path = " > ".join(hierarchy) if hierarchy else "—"
        score = r.get("_score", 0)
        text = r.get("text", "")

        # Truncate for display
        if len(text) > 600:
            text = text[:600] + "..."

        console.print(
            f"\n[bold][{i}] RHEL {r.get('version', '?')} — {r.get('guide_title', 'Unknown')}[/bold]"
        )
        console.print(f"    [dim]Section:[/dim] {section_path}")
        console.print(f"    [dim]Type:[/dim] {r.get('content_type', '?')} | [dim]Score:[/dim] {score:.4f}")
        console.print(f"    [dim]URL:[/dim] [link={r.get('url', '')}]{r.get('url', '')}[/link]")
        console.print()
        console.print(Markdown(text))


def _run_compare(
    searcher: HybridSearch,
    query: str,
    versions: list[str],
    top_n: int,
) -> None:
    """Run a version comparison and print grouped results."""
    console.print(f"\n[bold]Comparing across RHEL versions: {', '.join(versions)}[/bold]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    grouped = searcher.search_by_version(query, versions, top_n=top_n)

    for version in sorted(grouped.keys()):
        results = grouped[version]
        console.print(Rule(f"[bold cyan]RHEL {version}[/bold cyan]"))

        if not results:
            console.print("[yellow]No results for this version.[/yellow]")
            continue

        for i, r in enumerate(results[:3], start=1):
            hierarchy = r.get("section_hierarchy", [])
            if isinstance(hierarchy, str):
                import json
                try:
                    hierarchy = json.loads(hierarchy)
                except Exception:
                    hierarchy = []

            text = r.get("text", "")
            if len(text) > 400:
                text = text[:400] + "..."

            console.print(f"\n[bold][{i}] {r.get('guide_title', 'Unknown')}[/bold]")
            console.print(f"    [dim]{' > '.join(hierarchy) if hierarchy else '—'}[/dim]")
            console.print(f"    [link={r.get('url', '')}]{r.get('url', '')}[/link]")
            console.print()
            console.print(Markdown(text))

        console.print()


if __name__ == "__main__":
    app()
