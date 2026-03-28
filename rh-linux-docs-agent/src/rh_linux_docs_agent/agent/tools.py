"""
agent/tools.py — Tool functions that the agent calls to query documentation.

These are the "hands" of the agent — the functions it can call to retrieve
information from the vector database. pydantic-ai automatically makes these
available to the LLM via function calling.

When the agent needs information, it calls one of these tools:
- docs_search(): Find relevant documentation for a query
- docs_compare(): Compare how something works across RHEL versions

The tools return formatted strings that the agent reads and incorporates
into its answer.

Design notes:
- Tools are synchronous (pydantic-ai handles async internally)
- Each tool returns a formatted string ready for the LLM to read
- The format includes guide title, section path, text excerpt, and URL
- Results are truncated to avoid overwhelming the LLM context window
"""

import json
import logging
from typing import Any

from rh_linux_docs_agent.search.hybrid import HybridSearch
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)

# Global search instance — initialized once and reused for all tool calls.
# This avoids reloading the ~130MB embedding model on every tool call.
_searcher: HybridSearch | None = None


def get_searcher() -> HybridSearch:
    """
    Get the global HybridSearch instance, creating it if needed.

    Using a global here avoids reloading the embedding model (130MB) on
    every search request. The model loads in ~2 seconds on first call.

    Returns:
        The global HybridSearch instance.
    """
    global _searcher
    if _searcher is None:
        _searcher = HybridSearch()
    return _searcher


def docs_search(query: str, version: str | None = None) -> str:
    """
    Search RHEL documentation and return formatted results.

    This is the primary tool for the agent to find relevant documentation.
    The agent calls this with the user's question (or a reformulated version)
    and gets back the top matching documentation chunks.

    Args:
        query: The search query. Natural language or keywords both work.
               Examples:
               - "configure static IP address"
               - "nmcli connection modify"
               - "enable SELinux enforcing mode"
               - "dnf install package offline"
        version: Optional RHEL version filter: "8", "9", or "10".
                 If None, searches across all versions.

    Returns:
        Formatted string with top matching documentation chunks, each showing:
        - Version, guide title, section path
        - Relevance score
        - Content excerpt (first 800 chars)
        - Source URL for citation

    Example agent usage:
        result = docs_search("configure firewalld zone", version="9")
        # Returns formatted text with top 5 matching sections
    """
    searcher = get_searcher()

    try:
        results = searcher.search(query, version=version)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {e}\n\nMake sure the database has been populated by running: python scripts/ingest.py --version 9"

    if not results:
        version_msg = f" for RHEL {version}" if version else ""
        return (
            f"No results found{version_msg} for query: '{query}'\n\n"
            "Possible reasons:\n"
            "1. The database may be empty — run: python scripts/ingest.py --version 9\n"
            "2. Try different keywords or a broader query\n"
            f"3. The topic may not be covered in the indexed versions: {settings.supported_versions}"
        )

    return _format_search_results(results, query, version)


def docs_compare(query: str, versions: list[str] | None = None) -> str:
    """
    Compare documentation across multiple RHEL versions.

    Searches the same query across the specified versions and returns results
    grouped by version. This lets the agent identify how procedures, default
    values, or available tools differ between RHEL versions.

    Args:
        query: The topic to compare across versions.
               Examples:
               - "firewalld default zones"
               - "yum vs dnf package management"
               - "nftables iptables migration"
        versions: List of RHEL versions to compare.
                  Defaults to ["8", "9", "10"] if not specified.

    Returns:
        Formatted string with results grouped by version, showing how the
        documentation differs between versions.

    Example agent usage:
        result = docs_compare("network interface naming", ["8", "9"])
        # Returns results for RHEL 8 and RHEL 9 side by side
    """
    if not versions:
        versions = settings.supported_versions

    # Filter to only versions we support
    valid_versions = [v for v in versions if v in settings.supported_versions]
    if not valid_versions:
        return f"Invalid versions specified. Supported versions: {settings.supported_versions}"

    searcher = get_searcher()

    try:
        grouped_results = searcher.search_by_version(query, valid_versions)
    except Exception as e:
        logger.error(f"Compare search error: {e}")
        return f"Search error during version comparison: {e}"

    return _format_comparison_results(grouped_results, query)


def _format_search_results(
    results: list[dict[str, Any]],
    query: str,
    version: str | None,
) -> str:
    """
    Format search results into a readable string for the LLM.

    The format is designed to give the LLM enough context to generate
    a good answer: the source, section location, and content.

    Args:
        results: List of result dicts from HybridSearch.
        query: The original query (included in header for context).
        version: Version filter that was applied (for header).

    Returns:
        Multi-line formatted string.
    """
    version_str = f"RHEL {version}" if version else "all RHEL versions"
    lines = [
        f"=== Search Results for: '{query}' ({version_str}) ===",
        f"Found {len(results)} relevant documentation sections:",
        "",
    ]

    for i, r in enumerate(results, start=1):
        # Parse section hierarchy from JSON string if needed
        hierarchy = r.get("section_hierarchy", [])
        if isinstance(hierarchy, str):
            try:
                hierarchy = json.loads(hierarchy)
            except Exception:
                hierarchy = []

        section_path = " > ".join(hierarchy) if hierarchy else "—"
        score = r.get("_score", 0)
        text = r.get("text", "")

        # Truncate very long texts to avoid overwhelming the LLM context
        max_chars = 800
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        lines.extend([
            f"--- Result {i} ---",
            f"Version: RHEL {r.get('version', '?')}",
            f"Guide: {r.get('guide_title', 'Unknown')}",
            f"Section: {section_path}",
            f"Type: {r.get('content_type', '?')} | Score: {score:.4f}",
            f"URL: {r.get('url', '')}",
            "",
            text,
            "",
        ])

    return "\n".join(lines)


def _format_comparison_results(
    grouped: dict[str, list[dict[str, Any]]],
    query: str,
) -> str:
    """
    Format comparison results (grouped by version) for the LLM.

    Args:
        grouped: Dict of version → list of result dicts.
        query: The original comparison query.

    Returns:
        Multi-line formatted string with results by version.
    """
    lines = [
        f"=== Version Comparison: '{query}' ===",
        "",
    ]

    for version in sorted(grouped.keys()):
        results = grouped[version]
        lines.append(f"━━━ RHEL {version} ━━━")

        if not results:
            lines.append("  No documentation found for this version.")
            lines.append("")
            continue

        for i, r in enumerate(results[:3], start=1):  # Limit to top 3 per version
            hierarchy = r.get("section_hierarchy", [])
            if isinstance(hierarchy, str):
                try:
                    hierarchy = json.loads(hierarchy)
                except Exception:
                    hierarchy = []

            section_path = " > ".join(hierarchy) if hierarchy else "—"
            text = r.get("text", "")

            # Shorter truncation for comparison mode (more results to show)
            max_chars = 500
            if len(text) > max_chars:
                text = text[:max_chars] + "... [truncated]"

            lines.extend([
                f"  [{i}] {r.get('guide_title', 'Unknown')}",
                f"      Section: {section_path}",
                f"      URL: {r.get('url', '')}",
                f"      {text}",
                "",
            ])

    return "\n".join(lines)
