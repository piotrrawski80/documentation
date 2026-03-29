"""
test_api.py -- Tests for the FastAPI layer.

Uses FastAPI's TestClient (synchronous, no server needed).

Usage:
    python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
src = str(Path(__file__).resolve().parent.parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)

from fastapi.testclient import TestClient

from rh_linux_docs_agent.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a fully loaded QAEngine (once per module)."""
    app = create_app()

    # Force the startup event so the engine is loaded before tests run.
    # TestClient triggers startup/shutdown events via its context manager.
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_versions_indexed_is_list(self, client: TestClient):
        body = client.get("/health").json()
        assert isinstance(body["versions_indexed"], list)
        # We know RHEL 8 and 9 are indexed from the ingest runs
        assert "9" in body["versions_indexed"]


# ---------------------------------------------------------------------------
# GET /versions
# ---------------------------------------------------------------------------

class TestVersions:
    def test_returns_200(self, client: TestClient):
        r = client.get("/versions")
        assert r.status_code == 200

    def test_lists_all_supported_versions(self, client: TestClient):
        body = client.get("/versions").json()
        version_strs = [v["version"] for v in body["versions"]]
        assert "8" in version_strs
        assert "9" in version_strs
        assert "10" in version_strs

    def test_total_chunks_positive(self, client: TestClient):
        body = client.get("/versions").json()
        assert body["total_chunks"] > 0

    def test_rhel9_has_chunks(self, client: TestClient):
        body = client.get("/versions").json()
        rhel9 = [v for v in body["versions"] if v["version"] == "9"][0]
        assert rhel9["indexed"] is True
        assert rhel9["chunks"] > 0


# ---------------------------------------------------------------------------
# POST /ask — explicit version
# ---------------------------------------------------------------------------

class TestAskExplicit:
    def test_rhel9_firewalld(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to configure firewalld on RHEL 9?",
            "version": "9",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["resolved_version"] == "9"
        assert body["version_source"] == "explicit"
        assert body["confidence"] in ("high", "medium")
        assert len(body["sources"]) > 0
        assert body["latency_ms"] > 0

    def test_rhel8_selinux(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to troubleshoot SELinux denials?",
            "version": "8",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["resolved_version"] == "8"
        assert body["version_source"] == "explicit"
        assert body["confidence"] in ("high", "medium")

    def test_response_has_all_fields(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to install packages with dnf?",
            "version": "9",
        })
        body = r.json()
        for field in (
            "answer", "confidence", "coverage", "resolved_version",
            "version_source", "query_type", "intent", "sources", "latency_ms",
        ):
            assert field in body, f"Missing field: {field}"

    def test_source_structure(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to configure a static IP address?",
            "version": "9",
        })
        body = r.json()
        if body["sources"]:
            src = body["sources"][0]
            for field in ("title", "heading", "url", "guide_slug", "content_type", "rerank_score"):
                assert field in src, f"Missing source field: {field}"


# ---------------------------------------------------------------------------
# POST /ask — auto version
# ---------------------------------------------------------------------------

class TestAskAuto:
    def test_auto_detects_rhel9(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to configure firewalld on RHEL 9?",
            "version": "auto",
        })
        body = r.json()
        assert body["resolved_version"] == "9"
        assert body["version_source"] == "explicit"  # detected from query text

    def test_auto_detects_rhel8(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to configure networking on RHEL 8?",
            "version": "auto",
        })
        body = r.json()
        assert body["resolved_version"] == "8"
        assert body["version_source"] == "explicit"

    def test_auto_defaults_to_9(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to manage LVM volumes?",
            "version": "auto",
        })
        body = r.json()
        assert body["resolved_version"] == "9"
        assert body["version_source"] == "default"

    def test_auto_is_default_when_omitted(self, client: TestClient):
        """version field defaults to 'auto' when not provided."""
        r = client.post("/ask", json={"query": "How to enable FIPS?"})
        body = r.json()
        assert body["resolved_version"] == "9"
        assert body["version_source"] == "default"


# ---------------------------------------------------------------------------
# POST /ask — validation / edge cases
# ---------------------------------------------------------------------------

class TestAskValidation:
    def test_empty_query_rejected(self, client: TestClient):
        r = client.post("/ask", json={"query": "", "version": "9"})
        assert r.status_code == 422

    def test_invalid_version_rejected(self, client: TestClient):
        r = client.post("/ask", json={"query": "test", "version": "7"})
        assert r.status_code == 422

    def test_nonsense_query_low_confidence(self, client: TestClient):
        r = client.post("/ask", json={
            "query": "How to configure quantum entanglement routing?",
            "version": "9",
        })
        body = r.json()
        assert body["confidence"] in ("low", "insufficient")
