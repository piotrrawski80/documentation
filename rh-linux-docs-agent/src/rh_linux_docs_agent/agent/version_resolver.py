"""
agent/version_resolver.py — Detect RHEL version from user queries.

Extracts explicit version mentions from the query text.
If no version is found, defaults to RHEL 9.

Returns:
    resolved_version: "8", "9", or "10"
    version_source: "explicit" (mentioned in query) or "default" (not mentioned)

Examples:
    >>> resolve_version("How to configure firewalld on RHEL 8?")
    ('8', 'explicit')
    >>> resolve_version("How to configure firewalld?")
    ('9', 'default')
    >>> resolve_version("What changed in Red Hat Enterprise Linux 10?")
    ('10', 'explicit')
"""

import re

from rh_linux_docs_agent.config import settings

# ── Version detection patterns (ordered by specificity) ────────────────────

_VERSION_PATTERNS: list[re.Pattern] = [
    # "RHEL 9", "RHEL9", "rhel 9.4", "RHEL-9"
    re.compile(r"\brhel[\s\-]*(\d+)(?:\.\d+)?\b", re.I),
    # "Red Hat Enterprise Linux 9", "Red Hat Enterprise Linux 9.4"
    re.compile(r"\bred\s+hat\s+enterprise\s+linux\s+(\d+)(?:\.\d+)?\b", re.I),
    # "EL9", "el8" (common shorthand in sysadmin contexts)
    re.compile(r"\bel(\d+)\b", re.I),
]

DEFAULT_VERSION = "9"


def resolve_version(query: str) -> tuple[str, str]:
    """
    Detect RHEL version from query text.

    Scans for explicit version mentions using regex patterns.
    Only returns versions that are in settings.supported_versions.

    Args:
        query: User's natural language question.

    Returns:
        Tuple of (resolved_version, version_source).
        - resolved_version: "8", "9", or "10"
        - version_source: "explicit" or "default"
    """
    for pattern in _VERSION_PATTERNS:
        m = pattern.search(query)
        if m:
            v = m.group(1)
            if v in settings.supported_versions:
                return v, "explicit"

    return DEFAULT_VERSION, "default"
