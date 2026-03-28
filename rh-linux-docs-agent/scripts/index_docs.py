"""
scripts/index_docs.py — Embed chunked docs and store in LanceDB.

Reads JSON from data/chunked/rhel{version}/, embeds chunk_text using
BAAI/bge-small-en-v1.5, and stores everything in a local LanceDB table.

Usage:
    # Index all RHEL 9 chunks
    python scripts/index_docs.py --version 9

    # Index a single guide
    python scripts/index_docs.py --version 9 --guide configuring_and_managing_networking

    # Fresh start (drop existing data for this version first)
    python scripts/index_docs.py --version 9 --fresh

    # Show what's in the database
    python scripts/index_docs.py --stats

    # Custom paths
    python scripts/index_docs.py --version 9 --input data/chunked --db-path data/lancedb
"""

import sys
import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rh_linux_docs_agent.indexer.embedder import Embedder
from rh_linux_docs_agent.indexer.store import DocStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="index-docs",
    help="Embed chunked docs and store in LanceDB.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    version: str = typer.Option("9", "--version", "-v", help="RHEL version."),
    input_dir: Path = typer.Option(
        Path("data/chunked"), "--input", "-i", help="Chunked JSON root."
    ),
    db_path: Path = typer.Option(
        Path("data/lancedb"), "--db-path", "-d", help="LanceDB directory."
    ),
    guide: str = typer.Option(
        "", "--guide", "-g", help="Index only this guide slug."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Delete existing data for this version before indexing."
    ),
    stats: bool = typer.Option(
        False, "--stats", help="Show database stats and exit."
    ),
    build_index: bool = typer.Option(
        True,
        "--build-index/--no-build-index",
        help="Build vector + FTS indexes after insertion.",
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Embedding batch size."
    ),
) -> None:
    """Embed chunked section records and store in LanceDB."""

    start = time.time()
    store = DocStore(db_path=db_path)

    # ── Stats-only mode ───────────────────────────────────────────────────
    if stats:
        _show_stats(store)
        return

    # ── Discover chunk files ──────────────────────────────────────────────
    src = input_dir / f"rhel{version}"
    if not src.exists():
        console.print(
            f"[red]Error:[/red] {src} not found. Run chunk_docs.py first."
        )
        raise typer.Exit(1)

    json_files = sorted(
        f for f in src.iterdir()
        if f.suffix == ".json" and f.name.endswith(".chunks.json")
    )
    if guide:
        json_files = [f for f in json_files if f.stem.replace(".chunks", "") == guide]
        if not json_files:
            console.print(f"[red]Error:[/red] Guide '{guide}' not found in {src}.")
            raise typer.Exit(1)

    console.print(f"\n[bold]Indexing {len(json_files)} guides from {src}[/bold]")

    # ── Fresh mode: delete existing version data ──────────────────────────
    if fresh:
        n = store.delete_by_version(version)
        console.print(f"  Deleted {n} existing chunks for RHEL {version}")

    # ── Load all chunks ───────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for jf in json_files:
        chunks = json.loads(jf.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    console.print(f"  Loaded [bold]{len(all_chunks):,}[/bold] chunks from {len(json_files)} files")

    if not all_chunks:
        console.print("[yellow]No chunks to index.[/yellow]")
        return

    # ── Embed ─────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Embedding {len(all_chunks):,} chunks...[/bold]")
    embedder = Embedder()

    texts = [c["chunk_text"] for c in all_chunks]
    vectors = embedder.embed_with_progress(
        texts, description=f"Embedding RHEL {version}"
    )

    console.print(
        f"  Embedded [bold]{len(vectors):,}[/bold] chunks "
        f"(dim={len(vectors[0])})"
    )

    # ── Store in LanceDB ──────────────────────────────────────────────────
    console.print(f"\n[bold]Inserting into LanceDB at {db_path}...[/bold]")
    n_inserted = store.insert_chunks(all_chunks, vectors)
    console.print(f"  Inserted [bold]{n_inserted:,}[/bold] records")

    # ── Build indexes ─────────────────────────────────────────────────────
    if build_index:
        console.print(f"\n[bold]Building search indexes...[/bold]")
        store.create_indexes()
        console.print("  [green]Indexes built[/green]")

    # ── Summary ───────────────────────────────────────────────────────────
    _show_stats(store)
    console.print(f"\n[bold green]Done in {time.time() - start:.1f}s[/bold green]")


def _show_stats(store: DocStore) -> None:
    """Display database contents summary."""
    info = store.table_stats()
    total = info.get("total_rows", 0)

    console.print(f"\n[bold]LanceDB Stats:[/bold]")

    if total == 0:
        console.print("  [dim]Database is empty.[/dim]")
        return

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")

    t.add_row("Total chunks", f"{total:,}")
    t.add_row("Unique guides", f"{info.get('unique_guides', '?')}")
    t.add_row("", "")

    versions = info.get("versions", {})
    for v, count in sorted(versions.items()):
        t.add_row(f"  RHEL {v}", f"{count:,}")

    doc_types = info.get("doc_types", {})
    if doc_types:
        t.add_row("", "")
        for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
            t.add_row(f"  {dt}", f"{count:,}")

    console.print(t)


if __name__ == "__main__":
    app()
