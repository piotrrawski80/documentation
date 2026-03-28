"""
parser/models.py — Data classes for parsed RHEL documentation content.

These are the data structures produced by the HTML parser and consumed by
the chunker. Think of them as typed containers for the extracted content.

Flow:
  Raw HTML → Parser → [ParsedGuide, Section, ...] → Chunker → [Chunk, ...]

Schema version: 0.2.0
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Bump this when the output schema changes. Stored in every record
# so downstream consumers can detect stale data.
PARSER_VERSION = "0.2.0"


# ── doc_type classifier ──────────────────────────────────────────────────────

_DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"release_notes", "release_notes"),
    (r"security_hardening|using_selinux|managing_smart_card|crypto", "security"),
    (r"networking|network_file_services|firewalls|packet_filters|dns", "networking"),
    (r"storage|logical_volumes|file_systems|multipath|gfs2", "storage"),
    (r"installing|installation|upgrading|migrating|anaconda", "installation"),
    (r"identity_management|idm|trust_between|certificates_in_idm", "identity_management"),
    (r"containers|image_mode", "containers"),
    (r"virtualization", "virtualization"),
    (r"kernel|monitoring|performance|tuning", "system_management"),
    (r"ansible|system_roles|automating", "automation"),
    (r"developing|packaging|distributing|cpp_application", "development"),
    (r"web_console|cockpit", "web_console"),
    (r"selinux", "security"),
]


def classify_doc_type(slug: str) -> str:
    """
    Infer doc_type from guide slug using pattern matching.

    Returns one of: release_notes, security, networking, storage,
    installation, identity_management, containers, virtualization,
    system_management, automation, development, web_console, admin_guide.

    Falls back to 'admin_guide' when no pattern matches.
    """
    slug_lower = slug.lower()
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if re.search(pattern, slug_lower):
            return doc_type
    return "admin_guide"


def extract_minor_version(slug: str) -> str | None:
    """
    Extract minor version from release-notes slugs like '9.4_release_notes'.

    Returns '9.4' or None if not a versioned slug.
    """
    match = re.match(r"^(\d+\.\d+)_", slug)
    return match.group(1) if match else None


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Section:
    """
    A single section from a documentation guide.

    Attributes:
        heading: The section heading text, cleaned of UI artifacts.
        heading_level: The HTML heading level (2=chapter, 3=section, 4+=subsection).
        hierarchy: List of ancestor heading texts from top to current.
        body_text: The text content of this section (paragraphs, lists, code fences).
        code_blocks: List of verbatim code/command blocks, kept separate for chunking.
        content_type: Classification: "procedure", "concept", or "reference".
        section_id: HTML id attribute on the <section> element.
        has_tables: True if body_text contains a markdown table.
    """

    heading: str
    heading_level: int
    hierarchy: list[str]
    body_text: str
    code_blocks: list[str] = field(default_factory=list)
    content_type: str = "concept"
    section_id: str = ""
    has_tables: bool = False


@dataclass
class ParsedGuide:
    """
    All extracted content from one documentation guide.

    Attributes:
        slug: URL slug, e.g. "configuring_and_managing_networking".
        title: Human-readable guide title.
        version: RHEL major version string, e.g. "9".
        product: Always "rhel" for this parser.
        major_version: Same as version, e.g. "9".
        minor_version: e.g. "9.4" for release notes, None otherwise.
        doc_type: Inferred category (security, networking, admin_guide, ...).
        source_path: Local filesystem path to the source HTML file.
        guide_url: Reconstructed full URL to the guide on docs.redhat.com.
        parser_version: Version of this parser that produced the output.
        last_parsed_at: ISO 8601 timestamp of when parsing happened.
        sections: All sections extracted from the guide, in document order.
    """

    slug: str
    title: str
    version: str
    product: str
    major_version: str
    minor_version: str | None
    doc_type: str
    source_path: str
    guide_url: str
    parser_version: str
    last_parsed_at: str
    sections: list[Section] = field(default_factory=list)

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    def section_records(self) -> list[dict]:
        """
        Produce flat JSON-ready records — one per section — with all
        guide-level and section-level metadata merged.

        This is the primary output format consumed by chunking / indexing.
        """
        records = []
        for section in self.sections:
            record_id = _build_record_id(
                self.product, self.major_version, self.slug, section.section_id
            )
            section_url = self.guide_url
            if section.section_id:
                section_url = f"{self.guide_url}#{section.section_id}"
            heading_path_text = " > ".join(section.hierarchy)
            source_hash = hashlib.sha256(
                section.body_text.encode("utf-8")
            ).hexdigest()

            body = section.body_text
            char_count = len(body)
            word_count = len(body.split())

            records.append({
                # ── identifiers ──
                "record_id": record_id,
                # ── product metadata ──
                "product": self.product,
                "major_version": self.major_version,
                "minor_version": self.minor_version,
                "doc_type": self.doc_type,
                # ── guide metadata ──
                "guide_slug": self.slug,
                "guide_title": self.title,
                "version": self.version,
                "source_path": self.source_path,
                "guide_url": self.guide_url,
                "section_url": section_url,
                # ── section structure ──
                "heading": section.heading,
                "heading_level": section.heading_level,
                "hierarchy": section.hierarchy,
                "heading_path_text": heading_path_text,
                "section_id": section.section_id,
                # ── content ──
                "body_text": body,
                "code_blocks": section.code_blocks,
                "content_type": section.content_type,
                # ── helper fields ──
                "has_code_blocks": len(section.code_blocks) > 0,
                "has_tables": section.has_tables,
                "char_count": char_count,
                "word_count": word_count,
                # ── parsing metadata ──
                "source_hash": source_hash,
                "parser_version": self.parser_version,
                "last_parsed_at": self.last_parsed_at,
            })
        return records

    def to_dict(self) -> dict:
        """Full guide dict with nested section records."""
        return {
            "product": self.product,
            "major_version": self.major_version,
            "minor_version": self.minor_version,
            "doc_type": self.doc_type,
            "guide_slug": self.slug,
            "guide_title": self.title,
            "version": self.version,
            "source_path": self.source_path,
            "guide_url": self.guide_url,
            "parser_version": self.parser_version,
            "last_parsed_at": self.last_parsed_at,
            "total_sections": self.total_sections,
            "sections": self.section_records(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, output_path: Path) -> Path:
        """Save parsed guide as JSON. Creates parent dirs if needed."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path


def _build_record_id(
    product: str, major_version: str, slug: str, section_id: str
) -> str:
    """
    Build a stable, deterministic record ID.

    Format: {product}/{major_version}/{slug}/{section_id}
    Single separator for clean parsing. If section_id is empty,
    a short hash of the slug is used as fallback.
    """
    sid = section_id if section_id else hashlib.md5(slug.encode()).hexdigest()[:12]
    return f"{product}/{major_version}/{slug}/{sid}"


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
