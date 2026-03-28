"""
scripts/eval_retrieval.py — Retrieval quality evaluation for RHEL 9 docs.

Runs 20 test queries against the index and evaluates result relevance.
Compares three retrieval modes:
  1. vector-only
  2. hybrid (vector + BM25 + RRF)
  3. hybrid + cross-encoder reranking

For each query, shows top-5 results with scores, guides, headings.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --mode hybrid
    python scripts/eval_retrieval.py --mode vector
    python scripts/eval_retrieval.py --no-rerank
    python scripts/eval_retrieval.py --query 5          # Run only query #5
"""

import sys
import json
import time
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

console = Console()
app = typer.Typer(name="eval-retrieval", add_completion=False)


# ── 20 Test Queries with expected relevant guides ─────────────────────────────
# Each entry: (query, [expected_guide_slugs], category)
# expected_guide_slugs: guides that SHOULD appear in top-5 results
# category: query type for analysis

EVAL_SET = [
    # ── Networking (4 queries) ──
    (
        "How to configure a static IP address with nmcli on RHEL 9?",
        ["configuring_and_managing_networking"],
        "networking",
    ),
    (
        "Set up a network bridge for KVM virtual machines",
        ["configuring_and_managing_networking", "configuring_and_managing_virtualization"],
        "networking",
    ),
    (
        "Configure firewalld to allow only SSH and HTTPS",
        ["configuring_firewalls_and_packet_filters"],
        "networking",
    ),
    (
        "How to configure DNS client settings in resolv.conf",
        ["configuring_and_managing_networking"],
        "networking",
    ),
    # ── Security (4 queries) ──
    (
        "Enable SELinux enforcing mode and troubleshoot denials",
        ["using_selinux"],
        "security",
    ),
    (
        "Configure SELinux boolean for Apache httpd network access",
        ["using_selinux"],
        "security",
    ),
    (
        "Set up system-wide cryptographic policies FIPS mode",
        ["security_hardening"],
        "security",
    ),
    (
        "Configure SSH key-based authentication and disable password login",
        ["security_hardening", "configuring_and_managing_networking", "configuring_basic_system_settings"],
        "security",
    ),
    # ── Storage (3 queries) ──
    (
        "Create and extend LVM logical volumes",
        ["configuring_and_managing_logical_volumes"],
        "storage",
    ),
    (
        "How to configure Stratis storage pools on RHEL 9",
        ["managing_storage_devices", "managing_file_systems"],
        "storage",
    ),
    (
        "Mount NFS share persistently using /etc/fstab",
        ["configuring_and_using_network_file_services", "managing_file_systems"],
        "storage",
    ),
    # ── Containers (2 queries) ──
    (
        "Build and run a container image with Podman on RHEL 9",
        ["building_running_and_managing_containers"],
        "containers",
    ),
    (
        "Create a systemd service for a Podman container",
        ["building_running_and_managing_containers"],
        "containers",
    ),
    # ── System Management (3 queries) ──
    (
        "Configure automatic software updates with dnf-automatic",
        ["managing_software_with_the_dnf_tool"],
        "system_mgmt",
    ),
    (
        "How to use systemd journal to view and filter logs",
        ["monitoring_and_managing_system_status_and_performance", "configuring_basic_system_settings"],
        "system_mgmt",
    ),
    (
        "Configure kernel parameters with sysctl at boot time",
        ["monitoring_and_managing_system_status_and_performance", "managing_monitoring_and_updating_the_kernel"],
        "system_mgmt",
    ),
    # ── Identity Management (2 queries) ──
    (
        "Set up IdM (FreeIPA) server on RHEL 9",
        ["installing_identity_management"],
        "identity",
    ),
    (
        "Join RHEL client to Active Directory domain using SSSD",
        ["integrating_rhel_systems_directly_with_windows_active_directory", "configuring_and_using_network_file_services", "automating_system_administration_by_using_rhe"],
        "identity",
    ),
    # ── Virtualization + misc (2 queries) ──
    (
        "Create a KVM virtual machine using virt-install command",
        ["configuring_and_managing_virtualization"],
        "virtualization",
    ),
    (
        "What are the major changes in RHEL 9.4 release notes",
        ["9.4_release_notes"],
        "release_notes",
    ),
]


def _is_relevant(result: dict, expected_slugs: list[str]) -> bool:
    """Check if a result's guide_slug matches any expected slug."""
    slug = result.get("guide_slug", "")
    # Exact match or partial match (expected slug is a substring)
    for expected in expected_slugs:
        if expected in slug or slug in expected:
            return True
    return False


@app.command()
def main(
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, bm25"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip cross-encoder reranking"),
    query_num: int = typer.Option(0, "--query", "-q", help="Run only this query number (1-20)"),
    top_n: int = typer.Option(5, "--top-n", "-n", help="Results per query"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Retrieval candidates"),
) -> None:
    """Evaluate retrieval quality on 20 test queries for RHEL 9."""

    from rh_linux_docs_agent.search.retriever import Retriever

    console.print(f"\n[bold]Loading retrieval pipeline...[/bold]")
    console.print(f"  Mode: {mode} | Reranker: {'OFF' if no_rerank else 'ON'} | top_k={top_k} top_n={top_n}")

    retriever = Retriever(use_reranker=not no_rerank)

    console.print(f"  Pipeline ready.\n")

    queries = list(enumerate(EVAL_SET, start=1))
    if query_num:
        queries = [(i, e) for i, e in queries if i == query_num]
        if not queries:
            console.print(f"[red]Query #{query_num} not found (range: 1-{len(EVAL_SET)})[/red]")
            raise typer.Exit(1)

    # ── Run evaluation ────────────────────────────────────────────────────
    total_queries = len(queries)
    total_relevant_at_1 = 0
    total_relevant_at_3 = 0
    total_relevant_at_5 = 0
    total_reciprocal_rank = 0.0
    query_times: list[float] = []

    for idx, (query_text, expected_slugs, category) in queries:
        console.print(
            Panel(
                f"[bold cyan]Query {idx}/{len(EVAL_SET)}:[/bold cyan] {query_text}\n"
                f"[dim]Expected guides: {', '.join(expected_slugs)} | Category: {category}[/dim]",
                expand=False,
            )
        )

        t0 = time.time()
        results = retriever.retrieve(
            query_text,
            major_version="9",
            top_k=top_k,
            top_n=top_n,
            mode=mode,
        )
        elapsed = time.time() - t0
        query_times.append(elapsed)

        # Build results table
        t = Table(show_header=True, header_style="bold green", show_lines=True, width=140)
        t.add_column("#", width=3, style="dim")
        t.add_column("Rel", width=3, justify="center")
        t.add_column("Score", width=8, justify="right")
        t.add_column("Rerank", width=8, justify="right")
        t.add_column("Guide", width=32, no_wrap=True, overflow="ellipsis")
        t.add_column("Heading", width=45, no_wrap=True, overflow="ellipsis")
        t.add_column("Type", width=10)
        t.add_column("Preview", width=50, overflow="ellipsis")

        first_relevant_rank = 0
        relevant_in_top3 = False
        relevant_in_top5 = False

        for j, r in enumerate(results):
            is_rel = _is_relevant(r, expected_slugs)

            if is_rel and first_relevant_rank == 0:
                first_relevant_rank = j + 1
            if is_rel and j < 3:
                relevant_in_top3 = True
            if is_rel and j < 5:
                relevant_in_top5 = True

            rel_marker = "[bold green]Y[/bold green]" if is_rel else "[red]N[/red]"
            score_str = f"{r.get('_score', 0):.4f}"
            rerank_str = f"{r.get('_rerank_score', 0):.3f}" if '_rerank_score' in r else "-"
            preview = r.get("chunk_text", "")[:100].replace("\n", " ")

            t.add_row(
                str(j + 1),
                rel_marker,
                score_str,
                rerank_str,
                r.get("guide_slug", "")[:32],
                r.get("heading", "")[:45],
                r.get("content_type", ""),
                preview,
            )

        console.print(t)

        # Per-query metrics
        if first_relevant_rank == 1:
            total_relevant_at_1 += 1
        if relevant_in_top3:
            total_relevant_at_3 += 1
        if relevant_in_top5:
            total_relevant_at_5 += 1
        if first_relevant_rank > 0:
            total_reciprocal_rank += 1.0 / first_relevant_rank

        status = (
            f"[green]Hit@1[/green]" if first_relevant_rank == 1
            else f"[yellow]Hit@{first_relevant_rank}[/yellow]" if first_relevant_rank > 0
            else "[red]Miss[/red]"
        )
        console.print(f"  {status} | Time: {elapsed:.2f}s\n")

    # ── Summary metrics ───────────────────────────────────────────────────
    console.print(Panel("[bold]Evaluation Summary[/bold]", expand=False))

    st = Table(show_header=True, header_style="bold cyan")
    st.add_column("Metric", style="cyan", width=25)
    st.add_column("Value", justify="right", width=12)

    st.add_row("Total queries", str(total_queries))
    st.add_row("Mode", mode)
    st.add_row("Reranker", "OFF" if no_rerank else "ON")
    st.add_row("", "")
    st.add_row("Hit@1 (relevant at #1)", f"{total_relevant_at_1}/{total_queries} ({100*total_relevant_at_1/total_queries:.0f}%)")
    st.add_row("Hit@3 (relevant in top 3)", f"{total_relevant_at_3}/{total_queries} ({100*total_relevant_at_3/total_queries:.0f}%)")
    st.add_row("Hit@5 (relevant in top 5)", f"{total_relevant_at_5}/{total_queries} ({100*total_relevant_at_5/total_queries:.0f}%)")
    st.add_row("MRR (Mean Reciprocal Rank)", f"{total_reciprocal_rank/total_queries:.3f}")
    st.add_row("", "")
    st.add_row("Avg query time", f"{sum(query_times)/len(query_times):.2f}s")
    st.add_row("Total time", f"{sum(query_times):.1f}s")

    console.print(st)


if __name__ == "__main__":
    app()
