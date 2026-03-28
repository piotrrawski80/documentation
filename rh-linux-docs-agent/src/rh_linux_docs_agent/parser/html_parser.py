"""
html_parser.py — Extracts structured content from RHEL 9 HTML documentation.

RHEL 9 docs on docs.redhat.com use a modern HTML structure:
  - <div class="book"> is the root container
  - <section class="chapter"> for chapters (h2 headings)
  - <section class="section"> for sub-sections (h3+ headings)
  - <rh-alert class="admonition note/warning/..."> for admonitions
  - <pre class="language-plaintext"> for code blocks
  - Heading text contains "Copy link" spans that must be stripped

This parser walks the <section> tree recursively, which is more robust
than the flat heading-scan approach when dealing with nested content.

Usage:
    from rh_linux_docs_agent.parser.html_parser import parse_guide_html

    guide = parse_guide_html(
        html_path=Path("path/to/index.html"),
        slug="configuring_and_managing_networking",
        version="9",
    )
    for section in guide.sections:
        print(f"[{section.heading}]: {len(section.content)} chars")
"""

import re
import logging
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rh_linux_docs_agent.parser.models import (
    ParsedGuide,
    Section,
    PARSER_VERSION,
    classify_doc_type,
    extract_minor_version,
    now_iso,
)

logger = logging.getLogger(__name__)

# Base URL pattern for reconstructing source URLs
DOCS_BASE_URL = (
    "https://docs.redhat.com/en/documentation"
    "/red_hat_enterprise_linux/{version}/html-single/{slug}/index"
)

# Admonition types recognized in <rh-alert> elements
ADMONITION_TYPES = {
    "note": "NOTE",
    "warning": "WARNING",
    "important": "IMPORTANT",
    "tip": "TIP",
    "caution": "CAUTION",
}


def parse_guide_html(
    html_path: Path,
    slug: str,
    version: str,
    source_url: str = "",
) -> ParsedGuide:
    """
    Parse a single RHEL guide from a local HTML file.

    Args:
        html_path: Path to the local index.html file.
        slug: URL slug for the guide, e.g. "configuring_and_managing_networking".
        version: RHEL major version, e.g. "9".
        source_url: Optional URL override. If empty, reconstructed from slug+version.

    Returns:
        ParsedGuide with all extracted sections.
    """
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    title = _extract_title(soup)

    # Strip scripts, styles, nav
    _strip_noise(soup)

    # Find the book container
    book = soup.find("div", class_="book")
    if not book:
        book = soup.find("main") or soup.find("body") or soup
        logger.warning(f"No div.book found in {slug}, using fallback container")

    # Extract sections by walking the <section> tree
    sections = _extract_sections_recursive(book, version, slug)

    # Filter empties
    sections = [s for s in sections if s.body_text.strip() or s.code_blocks]

    logger.info(f"Parsed '{slug}': {len(sections)} sections")

    major_version = version
    minor_version = extract_minor_version(slug)
    doc_type = classify_doc_type(slug)

    if not source_url:
        source_url = DOCS_BASE_URL.format(version=version, slug=slug)

    return ParsedGuide(
        slug=slug,
        title=title,
        version=version,
        product="rhel",
        major_version=major_version,
        minor_version=minor_version,
        doc_type=doc_type,
        source_path=str(html_path),
        guide_url=source_url,
        parser_version=PARSER_VERSION,
        last_parsed_at=now_iso(),
        sections=sections,
    )


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract the guide title from the HTML <title> tag or titlepage."""
    # Strategy 1: titlepage h2.subtitle (RHEL 9 format: subtitle has the real title)
    titlepage = soup.find("div", class_="titlepage")
    if titlepage:
        subtitle = titlepage.find("h2", class_="subtitle")
        if subtitle:
            return _clean_heading_text(subtitle)
        h1 = titlepage.find("h1")
        if h1:
            return _clean_heading_text(h1)

    # Strategy 2: <title> tag (format: "Title | Product | Version | Red Hat")
    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        # Take only the first part before the pipe
        parts = text.split("|")
        if parts:
            return parts[0].strip()

    return "Unknown Guide"


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove scripts, styles, nav, and other non-content elements."""
    for tag in ["script", "style", "noscript", "nav", "header", "footer"]:
        for el in soup.find_all(tag):
            el.decompose()

    # Remove elements by known boilerplate classes
    boilerplate_classes = [
        "navigation", "navheader", "navfooter", "toc", "table-of-contents",
        "legal-notice", "legalnotice", "breadcrumb", "breadcrumbs",
        "footer", "feedback", "pager",
    ]
    for cls in boilerplate_classes:
        for el in soup.find_all(class_=re.compile(cls, re.IGNORECASE)):
            el.decompose()

    # Remove TOC/index by ID
    for el in soup.find_all(id=re.compile(r"^(toc|index|nav)", re.IGNORECASE)):
        el.decompose()


def _clean_heading_text(element: Tag) -> str:
    """
    Extract clean heading text from an element.

    RHEL 9 headings contain "Copy link" and "Link copied to clipboard!" spans
    that must be removed. Also normalizes whitespace.
    """
    # Remove copy-link UI spans
    for span in element.find_all("span", class_=["copy-link-text", "copy-link-text-confirmation"]):
        span.decompose()

    text = element.get_text(strip=True)
    # Normalize non-breaking spaces and multiple whitespace
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_sections_recursive(
    root: Tag,
    version: str,
    slug: str,
    hierarchy: list[str] | None = None,
) -> list[Section]:
    """
    Recursively walk <section> elements to extract content.

    The RHEL 9 HTML structure is:
      div.book
        section.chapter (h2 heading)
          p, div, pre, rh-alert, ... (chapter-level content)
          section.section (h3 heading)
            p, div, pre, ... (section content)
            section (h4 heading, deeper nesting)
              ...

    This recursive approach naturally handles arbitrary nesting depth.
    """
    if hierarchy is None:
        hierarchy = []

    sections: list[Section] = []

    # Find all chapter/section elements that are direct children
    section_elements = []
    for child in root.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "section":
            section_elements.append(child)
        elif child.name == "aside" and "chapter" in child.get("class", []):
            # Some "Additional resources" are <aside class="chapter">
            section_elements.append(child)

    for sec_el in section_elements:
        section_id = sec_el.get("id", "")

        # Find the heading (first h2/h3/h4/h5 in the titlepage or directly)
        heading_text, heading_level = _extract_section_heading(sec_el)

        if not heading_text:
            # Skip sections without headings (rare)
            continue

        current_hierarchy = hierarchy + [heading_text]

        # Collect this section's own content (not in child <section>s)
        content_parts: list[str] = []
        code_blocks: list[str] = []

        for child in sec_el.children:
            if not isinstance(child, Tag):
                continue
            # Skip child sections — they get their own Section objects
            if child.name == "section":
                continue
            # Skip the titlepage div (already extracted heading)
            if child.name == "div" and "titlepage" in child.get("class", []):
                continue
            # Extract content from this element
            _extract_element_content(child, content_parts, code_blocks)

        body_text = "\n\n".join(p for p in content_parts if p.strip())
        content_type = _classify_content(body_text)
        has_tables = bool(re.search(r"\|\s*---", body_text))

        sections.append(Section(
            heading=heading_text,
            heading_level=heading_level,
            hierarchy=current_hierarchy,
            body_text=body_text,
            code_blocks=code_blocks,
            content_type=content_type,
            section_id=section_id,
            has_tables=has_tables,
        ))

        # Recurse into child sections
        child_sections = _extract_sections_recursive(
            sec_el, version, slug, current_hierarchy
        )
        sections.extend(child_sections)

    return sections


def _extract_section_heading(section: Tag) -> tuple[str, int]:
    """
    Extract heading text and level from a <section> element.

    Returns:
        (heading_text, heading_level) or ("", 0) if no heading found.
    """
    # Check titlepage first
    titlepage = section.find("div", class_="titlepage", recursive=False)
    if titlepage:
        for level in range(2, 6):
            h = titlepage.find(f"h{level}")
            if h:
                return _clean_heading_text(h), level

    # Direct heading child
    for level in range(2, 6):
        h = section.find(f"h{level}", recursive=False)
        if h:
            return _clean_heading_text(h), level

    return "", 0


def _extract_element_content(
    element: Tag,
    content_parts: list[str],
    code_blocks: list[str],
) -> None:
    """
    Extract text content from a single HTML element and append to the lists.

    Handles: paragraphs, lists, code blocks, admonitions, tables, divs with lists.
    """
    tag = element.name
    classes = element.get("class", [])

    # ── Code blocks ──
    if tag == "pre":
        code_text = element.get_text()
        if code_text.strip():
            code_blocks.append(code_text.strip())
            content_parts.append(f"```\n{code_text.strip()}\n```")
        return

    # ── Admonitions (rh-alert) ──
    if tag == "rh-alert":
        label = _get_admonition_label(classes)
        text = _extract_admonition_text(element)
        if text:
            content_parts.append(f"{label}: {text}")
        return

    # ── Admonitions (div with admonition class — fallback) ──
    if tag == "div":
        adm_label = _get_admonition_label(classes)
        if adm_label:
            text = _extract_admonition_text(element)
            if text:
                content_parts.append(f"{adm_label}: {text}")
            return

    # ── Aside (additional resources, etc.) ──
    if tag == "aside":
        # Extract as a note-like block
        text_parts = []
        for child in element.children:
            if isinstance(child, Tag):
                if child.name in ("h2", "h3", "h4", "h5"):
                    text_parts.append(_clean_heading_text(child) + ":")
                elif child.name in ("ul", "ol"):
                    text_parts.append(_list_to_text(child))
                elif child.name == "p":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        text_parts.append(t)
                elif child.name == "div":
                    # div wrapping a list
                    inner_list = child.find(["ul", "ol"])
                    if inner_list:
                        text_parts.append(_list_to_text(inner_list))
        if text_parts:
            content_parts.append("\n".join(text_parts))
        return

    # ── Tables ──
    if tag == "table":
        table_text = _table_to_text(element)
        if table_text:
            content_parts.append(table_text)
        return

    # ── Paragraphs ──
    if tag == "p":
        text = element.get_text(separator=" ", strip=True)
        if text:
            content_parts.append(text)
        return

    # ── Lists (direct ul/ol) ──
    if tag in ("ul", "ol"):
        list_text = _list_to_text(element, code_blocks=code_blocks)
        if list_text:
            content_parts.append(list_text)
        return

    # ── Divs wrapping lists (div.itemizedlist, div.orderedlist) ──
    if tag == "div":
        # Check for wrapped lists
        inner_list = element.find(["ul", "ol"], recursive=False)
        if inner_list:
            list_text = _list_to_text(inner_list, code_blocks=code_blocks)
            if list_text:
                content_parts.append(list_text)
            return

        # Check for wrapped code
        inner_pre = element.find("pre", recursive=False)
        if inner_pre:
            code_text = inner_pre.get_text()
            if code_text.strip():
                code_blocks.append(code_text.strip())
                content_parts.append(f"```\n{code_text.strip()}\n```")
            return

        # Check for wrapped table
        inner_table = element.find("table", recursive=False)
        if inner_table:
            table_text = _table_to_text(inner_table)
            if table_text:
                content_parts.append(table_text)
            return

        # Generic div — recurse into children
        for child in element.children:
            if isinstance(child, Tag):
                _extract_element_content(child, content_parts, code_blocks)
        return

    # ── Custom elements (rh-code-block, etc.) — recurse into children ──
    if tag in ("rh-code-block",):
        inner_pre = element.find("pre")
        if inner_pre:
            code_text = inner_pre.get_text()
            if code_text.strip():
                code_blocks.append(code_text.strip())
                content_parts.append(f"```\n{code_text.strip()}\n```")
        return

    # ── Definition lists (dl) ──
    if tag == "dl":
        items = []
        current_term = ""
        for child in element.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "dt":
                current_term = child.get_text(separator=" ", strip=True)
            elif child.name == "dd":
                desc = child.get_text(separator=" ", strip=True)
                if current_term:
                    items.append(f"- {current_term}: {desc}")
                else:
                    items.append(f"- {desc}")
                current_term = ""
        if items:
            content_parts.append("\n".join(items))
        return

    # ── Fallback for unknown elements — recurse ──
    for child in element.children:
        if isinstance(child, Tag):
            _extract_element_content(child, content_parts, code_blocks)


def _get_admonition_label(classes: list[str]) -> str:
    """Return admonition label (NOTE, WARNING, etc.) from CSS classes, or empty string."""
    class_str = " ".join(classes).lower()
    for adm_class, label in ADMONITION_TYPES.items():
        if adm_class in class_str:
            return label
    return ""


def _extract_admonition_text(element: Tag) -> str:
    """Extract text from admonition, skipping the header div."""
    texts = []
    for child in element.children:
        if not isinstance(child, Tag):
            continue
        # Skip the header that just says "Note" or "Warning"
        if "admonition_header" in child.get("class", []):
            continue
        if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            continue
        text = child.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
    return " ".join(texts)


def _list_to_text(
    element: Tag,
    code_blocks: list[str] | None = None,
) -> str:
    """
    Convert a <ul> or <ol> to plain text with bullet/number markers.

    If code_blocks list is provided, code blocks found within list items
    are extracted and appended to it separately.
    """
    items: list[str] = []
    is_ordered = element.name == "ol"

    for i, li in enumerate(element.find_all("li", recursive=False)):
        # Extract code blocks from this list item before getting text
        li_code_blocks = li.find_all("pre")
        li_parts: list[str] = []

        for child in li.children:
            if not isinstance(child, Tag):
                text = str(child).strip()
                if text:
                    li_parts.append(text)
                continue

            # Code block inside list item
            if child.name == "pre" or (
                child.name in ("rh-code-block", "div")
                and child.find("pre")
            ):
                pre = child.find("pre") if child.name != "pre" else child
                if pre:
                    code_text = pre.get_text().strip()
                    if code_text:
                        if code_blocks is not None:
                            code_blocks.append(code_text)
                        li_parts.append(f"```\n{code_text}\n```")
                continue

            # Regular content
            text = child.get_text(separator=" ", strip=True)
            if text:
                li_parts.append(text)

        full_text = "\n".join(p for p in li_parts if p)
        if not full_text:
            continue

        prefix = f"{i + 1}." if is_ordered else "-"
        items.append(f"{prefix} {full_text}")

    return "\n".join(items)


def _table_to_text(element: Tag) -> str:
    """Convert an HTML table to a markdown-style pipe-delimited text table."""
    rows: list[list[str]] = []

    for tr in element.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"]):
            cell_text = re.sub(r"\s+", " ", cell.get_text(separator=" ", strip=True))
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    lines: list[str] = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _classify_content(content: str) -> str:
    """
    Classify section content as "procedure", "concept", or "reference".

    - procedure: numbered steps, imperative phrases
    - reference: tables, option/parameter lists
    - concept: everything else (default)
    """
    lower = content.lower()

    procedure_indicators = [
        r"^\d+\.\s",
        r"\bprocedure\b",
        r"\bsteps?:\s",
        r"\bprerequisites?\b",
        r"\bverification\b",
    ]
    for pattern in procedure_indicators:
        if re.search(pattern, lower, re.MULTILINE):
            return "procedure"

    reference_indicators = [
        r"\|\s*---",
        r"\bparameter\b",
        r"\bdefault\s+value\b",
        r"\bdescription\b.*\bvalue\b",
    ]
    for pattern in reference_indicators:
        if re.search(pattern, lower):
            return "reference"

    return "concept"
