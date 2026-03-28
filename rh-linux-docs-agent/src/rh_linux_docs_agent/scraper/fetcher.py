"""
fetcher.py — Downloads and caches RHEL documentation HTML pages.

This module downloads the "html-single" version of each RHEL documentation guide
and saves it to a local cache directory. Once downloaded, pages are never
re-downloaded unless you pass --force-refresh.

Why cache?
- Ingestion (parsing + embedding) may need to run multiple times during development
- Air-gapped environments need everything pre-downloaded
- Politeness to docs.redhat.com — don't hammer their servers
- The html-single version of a guide is one long HTML page with all chapters,
  which is much easier to parse than the multi-page version.

Cache layout:
  data/html_cache/
    9/
      configuring_networking.html
      managing_storage_devices.html
      ...
    10/
      ...
"""

import time
import logging
from pathlib import Path

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from rh_linux_docs_agent.config import settings
from rh_linux_docs_agent.scraper.discovery import GuideInfo, _fetch_with_retry

logger = logging.getLogger(__name__)


def fetch_guide(
    guide: GuideInfo,
    client: httpx.Client,
    force_refresh: bool = False,
) -> Path | None:
    """
    Download a single guide's HTML and save it to the local cache.

    If the file already exists in the cache and force_refresh is False,
    this function returns the cached path without making any network request.

    Args:
        guide: The GuideInfo object describing which guide to download.
        client: An httpx.Client already configured with headers and timeouts.
        force_refresh: If True, re-download even if the file is already cached.

    Returns:
        Path to the cached HTML file, or None if the download failed.

    Example:
        with httpx.Client(headers={"User-Agent": settings.user_agent}) as client:
            path = fetch_guide(guide, client)
            if path:
                html = path.read_text()
    """
    cache_path = _guide_cache_path(guide)

    # Check if already cached
    if cache_path.exists() and not force_refresh:
        logger.debug(f"Cache hit: {guide.slug}")
        return cache_path

    # Download the page
    logger.debug(f"Downloading: {guide.url}")
    html = _fetch_with_retry(client, guide.url)

    if html is None:
        logger.warning(f"Failed to download guide: {guide.slug} ({guide.url})")
        return None

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    logger.debug(f"Cached {guide.slug} ({len(html) / 1024:.1f} KB)")

    return cache_path


def fetch_all_guides(
    guides: list[GuideInfo],
    force_refresh: bool = False,
    show_progress: bool = True,
) -> dict[str, Path]:
    """
    Download all guides for a version, respecting rate limits and caching.

    Downloads guides one at a time with a delay between requests to be polite
    to docs.redhat.com. Already-cached guides are skipped instantly.

    Args:
        guides: List of GuideInfo objects to download.
        force_refresh: If True, re-download all guides even if cached.
        show_progress: If True, show a rich progress bar in the terminal.

    Returns:
        Dict mapping guide slug → Path to cached HTML file.
        Only includes guides that were successfully downloaded/found.

    Example:
        guides = discover_guides("9")
        paths = fetch_all_guides(guides)
        print(f"Downloaded {len(paths)} of {len(guides)} guides")
    """
    results: dict[str, Path] = {}
    total = len(guides)

    headers = {
        "User-Agent": settings.user_agent,
        # Pretend to be a browser to avoid being blocked
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    with httpx.Client(
        headers=headers,
        follow_redirects=True,
        timeout=60.0,  # Generous timeout — some guides are large (1MB+)
    ) as client:

        if show_progress:
            # Use rich's Progress bar for a nice terminal display
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.completed}/{task.total}"),
            ) as progress:
                task = progress.add_task(
                    f"Downloading RHEL {guides[0].version if guides else '?'} guides",
                    total=total,
                )
                results = _do_fetch_loop(
                    guides, client, force_refresh, progress=progress, task_id=task
                )
        else:
            results = _do_fetch_loop(guides, client, force_refresh)

    logger.info(
        f"Fetched {len(results)}/{total} guides "
        f"({'forced refresh' if force_refresh else 'with cache'})"
    )
    return results


def _do_fetch_loop(
    guides: list[GuideInfo],
    client: httpx.Client,
    force_refresh: bool,
    progress=None,
    task_id=None,
) -> dict[str, Path]:
    """
    Inner loop that processes each guide — download or cache hit.

    Args:
        guides: Guides to process.
        client: httpx client for HTTP requests.
        force_refresh: Whether to re-download cached guides.
        progress: Optional rich Progress instance for progress bar updates.
        task_id: The task ID in the Progress instance.

    Returns:
        Dict mapping slug → cached file Path.
    """
    results: dict[str, Path] = {}
    total = len(guides)

    for i, guide in enumerate(guides):
        is_cached = _guide_cache_path(guide).exists() and not force_refresh

        # Log progress with a counter like "[42/150]"
        status = "cached (skipped)" if is_cached else "downloading..."
        logger.info(f"[{i + 1}/{total}] {guide.slug} ... {status}")

        path = fetch_guide(guide, client, force_refresh=force_refresh)

        if path is not None:
            results[guide.slug] = path

        if progress is not None and task_id is not None:
            progress.advance(task_id)

        # Rate limiting — pause between requests to avoid being blocked.
        # We skip the delay if the guide was served from cache (no HTTP request made).
        if not is_cached and i < total - 1:
            time.sleep(settings.scrape_delay)

    return results


def _guide_cache_path(guide: GuideInfo) -> Path:
    """
    Compute the local cache file path for a given guide.

    Path format: data/html_cache/{version}/{slug}.html

    Args:
        guide: The guide whose cache path to compute.

    Returns:
        A Path object pointing to where this guide should be cached.
    """
    return settings.cache_dir / guide.version / f"{guide.slug}.html"


def list_cached_guides(version: str) -> list[str]:
    """
    List all guide slugs that have been cached for a given RHEL version.

    Useful for checking what's already downloaded before running ingestion.

    Args:
        version: The RHEL major version, e.g. "9".

    Returns:
        List of guide slugs (filenames without .html extension).

    Example:
        cached = list_cached_guides("9")
        print(f"Already have {len(cached)} guides for RHEL 9 in cache")
    """
    version_dir = settings.cache_dir / version
    if not version_dir.exists():
        return []

    return [p.stem for p in version_dir.glob("*.html")]


def get_cached_html(version: str, slug: str) -> str | None:
    """
    Read cached HTML for a specific guide, if it exists.

    Args:
        version: RHEL major version, e.g. "9".
        slug: Guide slug, e.g. "configuring_networking".

    Returns:
        The HTML content as a string, or None if not in cache.
    """
    cache_path = settings.cache_dir / version / f"{slug}.html"
    if not cache_path.exists():
        return None
    return cache_path.read_text(encoding="utf-8")
