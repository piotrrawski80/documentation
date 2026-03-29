"""
test_regression.py — Production regression test suite for QAEngine.

Runs 20+ core queries through the full pipeline and asserts:
  - Retrieval returns results
  - Confidence is reasonable
  - Answer mode is set
  - Key source guides/headings appear in results
  - No pydantic-ai imports in the pipeline
  - Interface intent detection works
  - Query classification works

Usage:
    python -m pytest tests/test_regression.py -v
    python -m pytest tests/test_regression.py -v -k "test_cli_lvm"
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
src = str(Path(__file__).resolve().parent.parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)

from rh_linux_docs_agent.agent.qa import (
    QAEngine, Answer, _extract_facets, _assess_confidence,
)
from rh_linux_docs_agent.search.retriever import classify_query, detect_interface_intent
from rh_linux_docs_agent.agent.version_resolver import resolve_version


# ── Shared engine fixture (loaded once per session) ──────────────────────────

@pytest.fixture(scope="session")
def engine():
    """Create a QAEngine once for all tests (model loading is expensive)."""
    return QAEngine(use_reranker=True)


def _ask(engine: QAEngine, query: str, major_version: str | None = None) -> Answer:
    """Ask a query. If major_version is None, version is auto-detected from query."""
    if major_version:
        return engine.ask(query, major_version=major_version)
    return engine.ask(query)


# ── Query classification tests ───────────────────────────────────────────────

class TestQueryClassification:
    def test_procedure(self):
        assert classify_query("How to configure a static IP?") == "procedure"

    def test_troubleshooting(self):
        assert classify_query("Troubleshoot SELinux denials") == "troubleshooting"

    def test_concept(self):
        assert classify_query("What is LVM?") == "concept"

    def test_reference(self):
        assert classify_query("List default firewalld zones") == "reference"


# ── Interface intent tests ───────────────────────────────────────────────────

class TestInterfaceIntent:
    def test_cli_command_line(self):
        assert detect_interface_intent("configure LVM from the command line") == "cli"

    def test_cli_nmcli(self):
        assert detect_interface_intent("set IP with nmcli") == "cli"

    def test_gui_web_console(self):
        assert detect_interface_intent("create LVM in the web console") == "gui"

    def test_gui_cockpit(self):
        assert detect_interface_intent("use cockpit to manage storage") == "gui"

    def test_neutral(self):
        assert detect_interface_intent("configure LVM on RHEL 9") == "neutral"


# ── Facet extraction tests ───────────────────────────────────────────────────

class TestFacetExtraction:
    def test_create_and_extend(self):
        facets = _extract_facets("create and extend LVM logical volumes")
        assert "create" in facets
        assert "extend" in facets

    def test_single_op(self):
        assert _extract_facets("configure a static IP") == []

    def test_install_and_start(self):
        facets = _extract_facets("install and start httpd")
        assert "install" in facets
        assert "start" in facets

    def test_enable_and_configure(self):
        facets = _extract_facets("enable and configure firewalld")
        assert "enable" in facets
        assert "configure" in facets


# ── Core production queries ──────────────────────────────────────────────────

class TestCoreQueries:
    """20 core production queries that must return meaningful results."""

    # ── Networking ───────────────────────────────────────────────────────

    def test_static_ip(self, engine):
        a = _ask(engine, "How do I configure a static IP address on RHEL 9?")
        assert a.context_chunks > 0
        assert a.confidence in ("high", "medium")
        assert any("nmcli" in s.chunk_text.lower() or "static" in s.heading.lower()
                    for s in a.sources)

    def test_firewalld(self, engine):
        a = _ask(engine, "How to configure firewalld to allow SSH and HTTPS on RHEL 9?")
        assert a.context_chunks > 0
        assert a.confidence in ("high", "medium")

    def test_dns_config(self, engine):
        a = _ask(engine, "How to configure DNS resolution on RHEL 9?")
        assert a.context_chunks > 0

    # ── Storage / LVM ────────────────────────────────────────────────────

    def test_cli_lvm_create_extend(self, engine):
        a = _ask(engine, "How do I create and extend an LVM logical volume from the command line on RHEL 9?")
        assert a.interface_intent == "cli"
        assert a.context_chunks >= 4
        assert a.answer_mode == "exact"
        # Must have chunks covering both create and extend
        headings = " ".join(s.heading for s in a.sources).lower()
        assert "creating" in headings or "create" in headings, \
            "At least one 'creating' chunk must appear in sources"
        assert "extending" in headings or "extend" in headings, \
            "At least one 'extending' chunk must appear in sources"

    def test_gui_lvm(self, engine):
        a = _ask(engine, "How to create LVM volumes in the web console on RHEL 9?")
        assert a.interface_intent == "gui"
        assert a.context_chunks > 0

    def test_lvm_snapshot(self, engine):
        a = _ask(engine, "How to create an LVM snapshot on RHEL 9?")
        assert a.context_chunks > 0

    # ── Security ─────────────────────────────────────────────────────────

    def test_selinux_troubleshoot(self, engine):
        a = _ask(engine, "Troubleshoot SELinux permission denials on RHEL 9")
        assert a.query_type == "troubleshooting"
        assert a.context_chunks > 0

    def test_fips(self, engine):
        a = _ask(engine, "How to enable FIPS crypto policy on RHEL 9?")
        assert a.context_chunks > 0
        assert a.confidence in ("high", "medium")

    def test_crypto_policies(self, engine):
        a = _ask(engine, "How to set system-wide crypto policies on RHEL 9?")
        assert a.context_chunks > 0

    # ── Containers ───────────────────────────────────────────────────────

    def test_podman_rootless(self, engine):
        a = _ask(engine, "How to run a rootless Podman container as a systemd service on RHEL 9?")
        assert a.context_chunks > 0

    def test_podman_build(self, engine):
        a = _ask(engine, "How to build a container image with Podman on RHEL 9?")
        assert a.context_chunks > 0

    # ── System administration ────────────────────────────────────────────

    def test_systemctl(self, engine):
        a = _ask(engine, "How to enable and start a systemd service on RHEL 9?")
        assert a.context_chunks > 0

    def test_dnf_install(self, engine):
        a = _ask(engine, "How to install packages with dnf on RHEL 9?")
        assert a.context_chunks > 0

    def test_user_management(self, engine):
        a = _ask(engine, "How to create a user account on RHEL 9?")
        assert a.context_chunks > 0

    def test_ssh_config(self, engine):
        a = _ask(engine, "How to configure SSH key-based authentication on RHEL 9?")
        assert a.context_chunks > 0

    # ── File systems ─────────────────────────────────────────────────────

    def test_xfs_filesystem(self, engine):
        a = _ask(engine, "How to create and mount an XFS filesystem on RHEL 9?")
        assert a.context_chunks > 0

    def test_nfs_mount(self, engine):
        a = _ask(engine, "How to mount an NFS share on RHEL 9?")
        assert a.context_chunks > 0

    # ── Performance / monitoring ─────────────────────────────────────────

    def test_journalctl(self, engine):
        a = _ask(engine, "How to view system logs with journalctl on RHEL 9?")
        assert a.context_chunks > 0

    def test_tuned(self, engine):
        a = _ask(engine, "How to configure performance tuning profiles on RHEL 9?")
        assert a.context_chunks > 0

    # ── Kernel / boot ────────────────────────────────────────────────────

    def test_grub(self, engine):
        a = _ask(engine, "How to modify kernel boot parameters on RHEL 9?")
        assert a.context_chunks > 0


# ── Negative / out-of-scope tests ────────────────────────────────────────────

class TestOutOfScope:
    def test_nonsense_query(self, engine):
        a = _ask(engine, "How to configure quantum entanglement routing on RHEL 9?")
        assert a.confidence == "insufficient"
        assert a.answer_mode == "insufficient"

    def test_wrong_product(self, engine):
        a = _ask(engine, "How to install Ubuntu desktop?")
        # Should return low confidence or insufficient
        assert a.confidence in ("low", "insufficient")


# ── Confidence logic tests ───────────────────────────────────────────────────

class TestConfidence:
    def test_mismatch_caps_at_low(self):
        conf, mode = _assess_confidence(
            [{"_rerank_score": 8.0}], "test", "mismatch detected", None,
        )
        assert conf == "low"
        assert mode == "partial"

    def test_empty_results(self):
        conf, mode = _assess_confidence([], "test", "", None)
        assert conf == "insufficient"
        assert mode == "insufficient"

    def test_strong_scores_no_mismatch(self):
        results = [
            {"_rerank_score": 8.0, "guide_slug": "g1", "content_type": "procedure"},
            {"_rerank_score": 7.0, "guide_slug": "g1", "content_type": "procedure"},
            {"_rerank_score": 6.0, "guide_slug": "g1", "content_type": "procedure"},
        ]
        conf, mode = _assess_confidence(results, "test", "", None)
        assert conf == "high"
        assert mode == "exact"


# ── No pydantic-ai dependency test ───────────────────────────────────────────

class TestNoPydanticAI:
    def test_qa_no_pydantic_ai_import(self):
        import rh_linux_docs_agent.agent.qa as qa_mod
        source = Path(qa_mod.__file__).read_text()
        assert "pydantic_ai" not in source
        assert "from pydantic_ai" not in source

    def test_app_no_pydantic_ai_import(self):
        import rh_linux_docs_agent.agent.app as app_mod
        source = Path(app_mod.__file__).read_text()
        assert "pydantic_ai" not in source


# ── Version resolver tests ──────────────────────────────────────────────────

class TestVersionResolver:
    """Test version detection from query text."""

    def test_explicit_rhel9(self):
        v, src = resolve_version("How to configure firewalld on RHEL 9?")
        assert v == "9"
        assert src == "explicit"

    def test_explicit_rhel8(self):
        v, src = resolve_version("How to configure firewalld on RHEL 8?")
        assert v == "8"
        assert src == "explicit"

    def test_explicit_rhel10(self):
        v, src = resolve_version("What's new in RHEL 10 for containers?")
        assert v == "10"
        assert src == "explicit"

    def test_explicit_no_space(self):
        v, src = resolve_version("Configure firewalld on RHEL9")
        assert v == "9"
        assert src == "explicit"

    def test_explicit_full_name(self):
        v, src = resolve_version(
            "How to enable FIPS on Red Hat Enterprise Linux 8?"
        )
        assert v == "8"
        assert src == "explicit"

    def test_explicit_el_shorthand(self):
        v, src = resolve_version("Install package on EL9")
        assert v == "9"
        assert src == "explicit"

    def test_default_no_version(self):
        v, src = resolve_version("How to configure firewalld?")
        assert v == "9"
        assert src == "default"

    def test_default_ambiguous(self):
        v, src = resolve_version("Best practices for LVM management")
        assert v == "9"
        assert src == "default"

    def test_explicit_with_minor_version(self):
        v, src = resolve_version("Changes in RHEL 9.4")
        assert v == "9"
        assert src == "explicit"

    def test_unsupported_version_defaults(self):
        v, src = resolve_version("Configure networking on RHEL 7")
        # 7 is not in supported_versions → default
        assert v == "9"
        assert src == "default"


# ── Multi-version QA tests ──────────────────────────────────────────────────

class TestMultiVersionQA:
    """Test that version resolver integrates with QAEngine correctly."""

    def test_rhel9_explicit_returns_results(self, engine):
        """RHEL 9 explicit query should return results (we have RHEL 9 data)."""
        a = _ask(engine, "How to configure firewalld on RHEL 9?")
        assert a.resolved_version == "9"
        assert a.version_source == "explicit"
        assert a.context_chunks > 0

    def test_rhel9_default_returns_results(self, engine):
        """No version mentioned → defaults to RHEL 9 → should return results."""
        a = _ask(engine, "How to configure firewalld?")
        assert a.resolved_version == "9"
        assert a.version_source == "default"
        assert a.context_chunks > 0

    def test_rhel8_explicit_no_data(self, engine):
        """RHEL 8 query but no RHEL 8 data indexed → insufficient or 0 chunks."""
        a = _ask(engine, "How to configure firewalld on RHEL 8?")
        assert a.resolved_version == "8"
        assert a.version_source == "explicit"
        # No RHEL 8 data indexed → should be insufficient
        # (unless legacy DB has RHEL 8 data)
        if a.context_chunks == 0:
            assert a.confidence == "insufficient"

    def test_rhel10_explicit_no_data(self, engine):
        """RHEL 10 query but no RHEL 10 data indexed → insufficient or 0 chunks."""
        a = _ask(engine, "What's new in RHEL 10 for containers?")
        assert a.resolved_version == "10"
        assert a.version_source == "explicit"
        if a.context_chunks == 0:
            assert a.confidence == "insufficient"

    def test_version_in_answer_metadata(self, engine):
        """Answer must carry resolved_version and version_source."""
        a = _ask(engine, "How to install packages with dnf on RHEL 9?")
        assert hasattr(a, "resolved_version")
        assert hasattr(a, "version_source")
        assert a.resolved_version in ("8", "9", "10")
        assert a.version_source in ("explicit", "default")

    def test_version_override(self, engine):
        """Explicit major_version parameter overrides auto-detection."""
        # Query says "RHEL 10" but we force version 9
        a = _ask(engine, "What's new in RHEL 10?", major_version="9")
        assert a.resolved_version == "9"
        assert a.context_chunks > 0  # Should find RHEL 9 data
