"""
scripts/search_demo.py — Run 5 example search queries against the LanceDB index.

Demonstrates vector similarity search on the embedded RHEL 9 documentation.
Each query is embedded using the same BGE model, then matched against stored chunks.

Usage:
    python scripts/search_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from rh_linux_docs_agent.indexer.embedder import Embedder
from rh_linux_docs_agent.indexer.store import DocStore

console = Console()

QUERIES = [
    "How to configure a static IP address with NetworkManager on RHEL 9?",
    "Set up firewalld to allow SSH and HTTP traffic",
    "Create and manage LVM logical volumes",
    "Configure SELinux boolean for httpd to connect to the network",
    "How to set up a Samba file share on RHEL 9?",
]


def main() -> None:
    console.print("\n[bold]Loading embedding model...[/bold]")
    embedder = Embedder()
    store = DocStore()

    total = store.get_total_count()
    console.print(f"  LanceDB: [bold]{total:,}[/bold] chunks indexed\n")

    for i, query in enumerate(QUERIES):
        console.print(Panel(f"[bold cyan]Query {i+1}:[/bold cyan] {query}", expand=False))

        query_vector = embedder.embed_query(query)
        results = store.search_vector(query_vector, limit=3, version_filter="9")

        t = Table(show_header=True, header_style="bold green", show_lines=True)
        t.add_column("#", width=3, style="dim")
        t.add_column("Score", width=7, justify="right")
        t.add_column("Guide", width=30, no_wrap=True, overflow="ellipsis")
        t.add_column("Heading", width=40, no_wrap=True, overflow="ellipsis")
        t.add_column("Type", width=10)
        t.add_column("Preview", width=60, overflow="ellipsis")

        for j, r in enumerate(results):
            dist = r.get("_distance", 0)
            score = f"{1 - dist:.3f}" if dist else "?"
            preview = r.get("chunk_text", "")[:150].replace("\n", " ")
            t.add_row(
                str(j + 1),
                score,
                r.get("guide_slug", "")[:30],
                r.get("heading", "")[:40],
                r.get("content_type", ""),
                preview,
            )

        console.print(t)
        console.print()


if __name__ == "__main__":
    main()
