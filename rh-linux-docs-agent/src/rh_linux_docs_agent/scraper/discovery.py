"""
discovery.py — Finds all documentation guide URLs for a given RHEL version.

This module visits the RHEL documentation landing page on docs.redhat.com and
extracts the list of available guides (books). Each guide covers a topic like
"Configuring networking" or "Managing storage devices".

The landing page URL pattern is:
  https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/{VERSION}

We then build single-page HTML URLs in the form:
  https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/{VERSION}/html-single/{guide_slug}/index
"""

import re
import time
import logging
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from rh_linux_docs_agent.config import settings

# Set up logging for this module. Log messages will show the module name.
logger = logging.getLogger(__name__)


@dataclass
class GuideInfo:
    """
    Metadata about a single documentation guide (book).

    Attributes:
        slug: The URL slug for this guide, e.g. "configuring_and_managing_networking".
              This is the part of the URL after /html-single/.
        title: Human-readable title, e.g. "Configuring and managing networking".
        version: The RHEL major version this guide belongs to, e.g. "9".
        url: The full single-page HTML URL for this guide.
    """

    slug: str
    title: str
    version: str
    url: str


def discover_guides(version: str, client: httpx.Client | None = None) -> list[GuideInfo]:
    """
    Discover all documentation guides available for a given RHEL version.

    Visits the RHEL docs landing page and parses the list of guide links.
    This gives us the complete list of guides to download and index.

    Args:
        version: RHEL major version string, e.g. "9" or "10".
        client: Optional httpx.Client to reuse (avoids creating a new one per call).
                If None, a temporary client is created.

    Returns:
        List of GuideInfo objects, one per available guide.
        Returns empty list if the landing page can't be fetched.

    Example:
        guides = discover_guides("9")
        print(f"Found {len(guides)} guides for RHEL 9")
        for g in guides[:3]:
            print(f"  {g.title}: {g.url}")
    """
    landing_url = f"{settings.docs_base_url}/{version}"
    logger.info(f"Discovering guides for RHEL {version} from {landing_url}")

    # Decide whether to use provided client or create a new temporary one
    own_client = client is None
    if own_client:
        client = httpx.Client(
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=30.0,
        )

    try:
        html = _fetch_with_retry(client, landing_url)
        if html is None:
            logger.error(f"Failed to fetch landing page for RHEL {version}")
            return []

        guides = _parse_guide_links(html, version)
        logger.info(f"Found {len(guides)} guides for RHEL {version}")
        return guides

    finally:
        # Only close the client if we created it here
        if own_client:
            client.close()


def _fetch_with_retry(client: httpx.Client, url: str) -> str | None:
    """
    Fetch a URL with automatic retries on failure.

    Retries on HTTP errors (429 Too Many Requests, 500, 503) with increasing
    wait times (exponential backoff: 2s, 4s, 8s between retries).

    Args:
        client: The httpx client to use for the request.
        url: The URL to fetch.

    Returns:
        The response body as a string, or None if all retries failed.
    """
    for attempt in range(settings.scrape_retries + 1):
        try:
            response = client.get(url)

            # 200 OK — success
            if response.status_code == 200:
                return response.text

            # 429 Too Many Requests or server error — wait and retry
            if response.status_code in (429, 500, 502, 503, 504):
                wait_seconds = 2 ** attempt  # 1s, 2s, 4s, 8s
                logger.warning(
                    f"HTTP {response.status_code} for {url} "
                    f"(attempt {attempt + 1}/{settings.scrape_retries + 1}). "
                    f"Waiting {wait_seconds}s before retry..."
                )
                time.sleep(wait_seconds)
                continue

            # 404 Not Found or other errors — don't retry
            logger.error(f"HTTP {response.status_code} for {url} — skipping")
            return None

        except httpx.RequestError as e:
            # Network error (connection refused, DNS failure, timeout, etc.)
            wait_seconds = 2 ** attempt
            logger.warning(
                f"Network error fetching {url}: {e} "
                f"(attempt {attempt + 1}/{settings.scrape_retries + 1}). "
                f"Waiting {wait_seconds}s before retry..."
            )
            time.sleep(wait_seconds)

    logger.error(f"All {settings.scrape_retries + 1} attempts failed for {url}")
    return None


def _parse_guide_links(html: str, version: str) -> list[GuideInfo]:
    """
    Parse the landing page HTML and extract all guide links.

    Looks for links pointing to /html-single/ URLs, which are the single-page
    HTML versions of each guide (one long page with the full book content).

    Args:
        html: The HTML content of the RHEL docs landing page.
        version: The RHEL version being parsed (used to tag GuideInfo objects).

    Returns:
        List of GuideInfo objects for each discovered guide.
    """
    soup = BeautifulSoup(html, "lxml")
    guides: list[GuideInfo] = []
    seen_slugs: set[str] = set()  # Deduplicate in case the same guide appears multiple times

    # Find all anchor tags on the page
    for link in soup.find_all("a", href=True):
        href = link["href"]

        # We want links like:
        #   /en/documentation/red_hat_enterprise_linux/9/html-single/configuring_networking/index
        # or full URLs:
        #   https://docs.redhat.com/en/documentation/.../html-single/configuring_networking/index

        # Normalize to just the path portion
        if href.startswith("http"):
            # Strip the domain to get just the path
            match = re.search(r"docs\.redhat\.com(/.*)", href)
            if match:
                href = match.group(1)

        # Match the html-single path pattern
        pattern = rf"/en/documentation/red_hat_enterprise_linux/{re.escape(version)}/html-single/([^/]+)"
        match = re.search(pattern, href)
        if not match:
            continue

        slug = match.group(1)

        # Skip if we've already seen this slug (duplicate links)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        # Get the link text as the guide title
        title = link.get_text(strip=True)

        # Skip empty titles or navigation-only links
        if not title or len(title) < 3:
            continue

        # Build the canonical single-page HTML URL
        url = f"{settings.docs_base_url}/{version}/html-single/{slug}/index"

        guides.append(GuideInfo(slug=slug, title=title, version=version, url=url))

    # If we found nothing with the strict pattern, try a looser approach
    # (the landing page structure can vary between RHEL versions)
    if not guides:
        logger.warning(
            f"No guides found with primary pattern for RHEL {version}. "
            "Trying fallback discovery..."
        )
        guides = _fallback_discovery(soup, version)

    return guides


def _fallback_discovery(soup: BeautifulSoup, version: str) -> list[GuideInfo]:
    """
    Fallback guide discovery when the primary pattern finds nothing.

    Some RHEL version pages have different HTML structures. This tries a broader
    search for any links that look like documentation guide links.

    Args:
        soup: BeautifulSoup-parsed landing page.
        version: The RHEL version string.

    Returns:
        List of GuideInfo objects found with the fallback method.
    """
    guides: list[GuideInfo] = []
    seen_slugs: set[str] = set()

    for link in soup.find_all("a", href=True):
        href = str(link["href"])

        # Look for any /html-single/ link that might be a guide
        if "/html-single/" not in href:
            continue

        # Extract the slug — the part right after /html-single/
        parts = href.split("/html-single/")
        if len(parts) < 2:
            continue

        slug_part = parts[1].split("/")[0].strip()
        if not slug_part or slug_part in seen_slugs:
            continue
        seen_slugs.add(slug_part)

        title = link.get_text(strip=True) or slug_part.replace("_", " ").title()

        url = f"{settings.docs_base_url}/{version}/html-single/{slug_part}/index"
        guides.append(GuideInfo(slug=slug_part, title=title, version=version, url=url))

    return guides
