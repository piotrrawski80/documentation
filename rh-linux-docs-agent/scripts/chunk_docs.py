"""
scripts/chunk_docs.py — Chunk parsed RHEL section records into embedding-ready pieces.

Reads JSON from data/parsed/rhel{version}/, splits large sections at semantic
boundaries, and writes chunked JSON to data/chunked/rhel{version}/.

Usage:
    # Chunk all parsed RHEL 9 guides
    python scripts/chunk_docs.py --version 9

    # Chunk a single guide
    python scripts/chunk_docs.py --version 9 --guide security_hardening

    # Stats only (no file output)
    python scripts/chunk_docs.py --version 9 --stats-only

    # Custom paths
    python scripts/chunk_docs.py --version 9 --input data/parsed --output data/chunked
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

from rh_linux_docs_agent.chunker.splitter import chunk_section_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(name="chunk-docs", help="Chunk parsed docs into embedding-ready pieces.", add_completion=False)
console = Console()


@app.command()
def main(
    version: str = typer.Option("9", "--version", "-v", help="RHEL version."),
    input_dir: Path = typer.Option(Path("data/parsed"), "--input", "-i", help="Parsed JSON root."),
    output_dir: Path = typer.Option(Path("data/chunked"), "--output", "-o", help="Chunked JSON root."),
    guide: str = typer.Option("", "--guide", "-g", help="Chunk only this guide slug."),
    stats_only: bool = typer.Option(False, "--stats-only", help="Show stats without writing files."),
) -> None:
    """Chunk parsed section records into embedding-ready pieces."""

    start = time.time()
    src = input_dir / f"rhel{version}"
    if not src.exists():
        console.print(f"[red]Error:[/red] {src} not found. Run parse_docs.py first.")
        raise typer.Exit(1)

    # Discover guide JSON files
    json_files = sorted(f for f in src.iterdir() if f.suffix == ".json" and f.name != "_manifest.json")
    if guide:
        json_files = [f for f in json_files if f.stem == guide]
        if not json_files:
            console.print(f"[red]Error:[/red] Guide '{guide}' not found in {src}.")
            raise typer.Exit(1)

    console.print(f"\n[bold]Chunking {len(json_files)} guides from {src}[/bold]")

    all_chunks: list[dict] = []
    guide_stats: list[dict] = []

    for i, jf in enumerate(json_files):
        data = json.loads(jf.read_text(encoding="utf-8"))
        records = data.get("sections", [])

        guide_chunks: list[dict] = []
        for rec in records:
            chunks = chunk_section_record(rec)
            guide_chunks.extend(chunks)

        all_chunks.extend(guide_chunks)
        slug = data.get("guide_slug", jf.stem)
        n_rec = len(records)
        n_ch = len(guide_chunks)
        ratio = f"{n_ch / n_rec:.1f}x" if n_rec else "0"

        guide_stats.append({"slug": slug, "sections": n_rec, "chunks": n_ch})
        console.print(f"  [{i+1}/{len(json_files)}] {slug}: {n_rec} sections -> {n_ch} chunks ({ratio})")

    # ── Write output ──────────────────────────────────────────────────────
    if not stats_only and all_chunks:
        dst = output_dir / f"rhel{version}"
        dst.mkdir(parents=True, exist_ok=True)

        # Group chunks by guide_slug for per-guide files
        by_guide: dict[str, list[dict]] = {}
        for ch in all_chunks:
            by_guide.setdefault(ch["guide_slug"], []).append(ch)

        for slug, chunks in by_guide.items():
            out_file = dst / f"{slug}.chunks.json"
            out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        # Manifest
        from rh_linux_docs_agent.parser.models import PARSER_VERSION, now_iso
        manifest = {
            "product": "rhel",
            "major_version": version,
            "parser_version": PARSER_VERSION,
            "generated_at": now_iso(),
            "total_guides": len(json_files),
            "total_chunks": len(all_chunks),
            "guides": guide_stats,
        }
        (dst / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n  Saved {len(by_guide)} chunk files + manifest to {dst}/")

    # ── Summary stats ─────────────────────────────────────────────────────
    _show_stats(all_chunks, guide_stats)

    console.print(f"\n[bold green]Done in {time.time()-start:.1f}s[/bold green]")


def _show_stats(chunks: list[dict], guide_stats: list[dict]) -> None:
    if not chunks:
        console.print("[yellow]No chunks produced.[/yellow]")
        return

    total = len(chunks)
    with_code = sum(1 for c in chunks if c["has_code_blocks"])
    with_tables = sum(1 for c in chunks if c["has_tables"])
    empty = sum(1 for c in chunks if c["char_count"] == 0)
    total_chars = sum(c["char_count"] for c in chunks)
    total_words = sum(c["word_count"] for c in chunks)
    char_counts = sorted(c["char_count"] for c in chunks)
    total_sections = sum(g["sections"] for g in guide_stats)

    # content_type breakdown
    ct: dict[str, int] = {}
    for c in chunks:
        ct[c["content_type"]] = ct.get(c["content_type"], 0) + 1

    # chunk_index distribution (how many sections needed splitting)
    max_idx = max(c["chunk_index"] for c in chunks)
    single_chunk = sum(1 for c in chunks if c["chunk_index"] == 0
                       and not any(c2["parent_record_id"] == c["parent_record_id"]
                                   and c2["chunk_index"] > 0 for c2 in chunks))
    # faster: count unique parent_record_ids that produced >1 chunk
    from collections import Counter
    parent_counts = Counter(c["parent_record_id"] for c in chunks)
    multi_chunk_sections = sum(1 for v in parent_counts.values() if v > 1)
    single_chunk_sections = sum(1 for v in parent_counts.values() if v == 1)

    console.print(f"\n[bold]Chunk Summary:[/bold]")

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")

    t.add_row("Input sections", f"{total_sections:,}")
    t.add_row("Output chunks", f"{total:,}")
    t.add_row("Expansion ratio", f"{total/max(total_sections,1):.2f}x")
    t.add_row("Sections kept as 1 chunk", f"{single_chunk_sections:,}")
    t.add_row("Sections split into >1", f"{multi_chunk_sections:,}")
    t.add_row("", "")
    t.add_row("Chunks with code_blocks", f"{with_code:,}")
    t.add_row("Chunks with tables", f"{with_tables:,}")
    t.add_row("Empty chunks (0 chars)", f"{empty}")
    t.add_row("", "")
    t.add_row("Total chars", f"{total_chars:,}")
    t.add_row("Total words", f"{total_words:,}")
    t.add_row("Avg chars/chunk", f"{total_chars // total:,}")
    t.add_row("Median chars/chunk", f"{char_counts[total//2]:,}")
    t.add_row("Min chars", f"{char_counts[0]:,}")
    t.add_row("Max chars", f"{char_counts[-1]:,}")
    t.add_row("p10 chars", f"{char_counts[total//10]:,}")
    t.add_row("p90 chars", f"{char_counts[total*9//10]:,}")
    t.add_row("", "")
    for key in ("procedure", "concept", "reference"):
        t.add_row(f"  {key}", f"{ct.get(key, 0):,}")

    console.print(t)

    # Top 10 largest chunks
    top = sorted(chunks, key=lambda c: c["char_count"], reverse=True)[:10]
    console.print(f"\n[bold]Top 10 largest chunks:[/bold]")
    lt = Table(show_header=True, header_style="bold cyan")
    lt.add_column("#", style="dim", width=4)
    lt.add_column("chars", justify="right", width=9)
    lt.add_column("idx", justify="right", width=4)
    lt.add_column("guide_slug", width=35, no_wrap=True, overflow="ellipsis")
    lt.add_column("heading", no_wrap=True, overflow="ellipsis")
    for i, c in enumerate(top):
        lt.add_row(str(i+1), f"{c['char_count']:,}", str(c["chunk_index"]),
                   c["guide_slug"][:35], c["heading"][:55])
    console.print(lt)

    # Top 10 most-split sections
    most_split = parent_counts.most_common(10)
    console.print(f"\n[bold]Top 10 most-split sections:[/bold]")
    st = Table(show_header=True, header_style="bold cyan")
    st.add_column("#", style="dim", width=4)
    st.add_column("chunks", justify="right", width=7)
    st.add_column("parent_record_id", no_wrap=True, overflow="ellipsis")
    for i, (pid, cnt) in enumerate(most_split):
        st.add_row(str(i+1), str(cnt), pid[:90])
    console.print(st)


if __name__ == "__main__":
    app()
