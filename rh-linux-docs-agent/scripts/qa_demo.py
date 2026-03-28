"""
scripts/qa_demo.py — Demonstrate the QA pipeline with example queries.

Runs queries through the full pipeline: retrieve → rerank → generate answer.
Shows the answer, sources, confidence level, and timing.

Usage:
    python scripts/qa_demo.py
    python scripts/qa_demo.py --query "How to configure firewalld?"
    python scripts/qa_demo.py --no-rerank
    python scripts/qa_demo.py --context-only   # Show context without LLM
"""

import sys
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

console = Console()
app = typer.Typer(name="qa-demo", add_completion=False)

DEMO_QUERIES = [
    # Q1: Procedural, clear answer expected
    "How do I create and extend an LVM logical volume on RHEL 9?",
    # Q2: Configuration with code blocks
    "How to configure firewalld to allow only SSH and HTTPS traffic on RHEL 9?",
    # Q3: SELinux troubleshooting
    "How do I troubleshoot SELinux denials for Apache httpd on RHEL 9?",
    # Q4: Container-related
    "How to create a systemd service for a rootless Podman container on RHEL 9?",
    # Q5: Deliberately vague / insufficient evidence query
    "How to configure Oracle Database RAC clustering on RHEL 9?",
    # Q6: Multi-source — should pull from multiple guides
    "How to join a RHEL 9 system to an Active Directory domain?",
]


@app.command()
def main(
    query: str = typer.Option("", "--query", "-q", help="Run a single custom query"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip reranking"),
    context_only: bool = typer.Option(False, "--context-only", help="Show context, skip LLM"),
) -> None:
    """Run QA demo queries against the RHEL 9 documentation index."""

    from rh_linux_docs_agent.agent.qa import QAEngine

    console.print("\n[bold]Loading QA pipeline...[/bold]")
    engine = QAEngine(use_reranker=not no_rerank)
    console.print("  Pipeline ready.\n")

    queries = [query] if query else DEMO_QUERIES

    for i, q in enumerate(queries, start=1):
        console.print(Panel(
            f"[bold cyan]Query {i}/{len(queries)}:[/bold cyan] {q}",
            expand=False,
        ))

        if context_only:
            results, context, sources = engine.retrieve_only(q, major_version="9")
            console.print(f"\n[dim]Retrieved {len(results)} chunks[/dim]\n")
            for j, s in enumerate(sources, start=1):
                console.print(f"  [bold][{j}][/bold] {s.guide_title}")
                console.print(f"      {s.heading}")
                console.print(f"      [dim]{s.section_url}[/dim]")
                console.print(f"      score={s.rerank_score:.3f}  type={s.content_type}")
                console.print()
            continue

        answer = engine.ask(q, major_version="9")

        # Confidence banner
        conf_color = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
            "insufficient": "red bold",
        }.get(answer.confidence, "white")
        console.print(f"  Confidence: [{conf_color}]{answer.confidence.upper()}[/{conf_color}]")
        console.print(f"  Query type: [cyan]{answer.query_type}[/cyan]")
        console.print(f"  Retrieval: {answer.retrieval_time_s}s | Generation: {answer.generation_time_s}s | Model: {answer.model}")
        console.print(f"  Context chunks: {answer.context_chunks}\n")

        # Answer
        border = "green" if answer.confidence == "high" else "yellow"
        console.print(Panel(
            answer.text,
            title="Answer",
            border_style=border,
        ))

        # Sources table
        if answer.sources:
            st = Table(show_header=True, header_style="bold cyan", show_lines=True)
            st.add_column("#", width=3)
            st.add_column("Guide", width=35, overflow="ellipsis")
            st.add_column("Heading", width=45, overflow="ellipsis")
            st.add_column("Type", width=10)
            st.add_column("URL", width=70, overflow="ellipsis")

            for j, s in enumerate(answer.sources, start=1):
                st.add_row(
                    str(j),
                    s.guide_title[:35],
                    s.heading[:45],
                    s.content_type,
                    s.section_url,
                )
            console.print(st)

        console.print()


if __name__ == "__main__":
    app()
