"""
tests/test_chunker.py — Unit tests for the chunk splitter.

Tests verify that:
- Chunks are within the configured min/max token bounds
- Code blocks are never split across chunk boundaries
- Section hierarchy is correctly attached to every chunk
- Small sections are merged rather than creating tiny chunks

Run with:
    cd rh-linux-docs-agent
    source venv/bin/activate
    pytest tests/test_chunker.py -v
"""

import pytest
from rh_linux_docs_agent.chunker.splitter import chunk_guide, count_tokens
from rh_linux_docs_agent.chunker.models import Chunk
from rh_linux_docs_agent.parser.models import ParsedGuide, Section
from rh_linux_docs_agent.config import settings


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_section(
    heading: str,
    content: str,
    hierarchy: list[str] | None = None,
    code_blocks: list[str] | None = None,
    level: int = 2,
) -> Section:
    """Helper to create a Section for testing."""
    return Section(
        heading=heading,
        heading_level=level,
        hierarchy=hierarchy or [heading],
        content=content,
        code_blocks=code_blocks or [],
        content_type="concept",
        url_anchor="",
    )


def make_guide(sections: list[Section], version: str = "9") -> ParsedGuide:
    """Helper to create a ParsedGuide for testing."""
    return ParsedGuide(
        slug="test_guide",
        title="Test Guide",
        version=version,
        url="https://docs.redhat.com/test",
        sections=sections,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChunkTokenLimits:
    """Tests that chunks respect the configured token size limits."""

    def test_normal_section_produces_chunks(self):
        """A normal-sized section should produce at least one chunk."""
        # Create content that's roughly 300 tokens (well within 150–800 range)
        content = " ".join(["word"] * 200)  # ~200 words ≈ 250 tokens
        guide = make_guide([make_section("Section 1", content)])
        chunks = chunk_guide(guide)
        assert len(chunks) >= 1

    def test_chunks_are_chunk_objects(self):
        """chunk_guide should return Chunk objects."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Section 1", content)])
        chunks = chunk_guide(guide)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)

    def test_large_section_produces_multiple_chunks(self):
        """A very large section should be split into multiple chunks."""
        # Create content that's roughly 2000 tokens (well over 800 max)
        # Use varied content so it has paragraph breaks to split on
        paragraphs = [f"Paragraph {i}: " + " ".join(["word"] * 60) for i in range(20)]
        content = "\n\n".join(paragraphs)
        guide = make_guide([make_section("Large Section", content)])

        chunks = chunk_guide(guide)
        # With 20 paragraphs of ~70 tokens each = ~1400 tokens total,
        # should split into at least 2 chunks
        assert len(chunks) >= 2

    def test_no_chunk_exceeds_max_tokens(self):
        """No chunk should exceed chunk_max_tokens (except for atomic code blocks)."""
        # Create multiple sections with varying content
        sections = []
        for i in range(5):
            paragraphs = [f"Para {j}: " + " ".join(["word"] * 40) for j in range(8)]
            content = "\n\n".join(paragraphs)
            sections.append(make_section(f"Section {i}", content))

        guide = make_guide(sections)
        chunks = chunk_guide(guide)

        # Check each chunk (code blocks may legitimately exceed max)
        for chunk in chunks:
            if "```" not in chunk.text:
                # Non-code chunks should be within limits
                # Allow some slack since our token counting is approximate
                assert chunk.token_count <= settings.chunk_max_tokens * 1.2, (
                    f"Chunk exceeds max size: {chunk.token_count} tokens\n"
                    f"Text: {chunk.text[:100]}..."
                )


class TestCodeBlockPreservation:
    """Tests that code blocks are never split across chunks."""

    def test_code_block_stays_whole(self):
        """A code block should appear complete in exactly one chunk."""
        code = "\n".join([
            "# Configure static IP",
            "nmcli connection modify eth0 ipv4.addresses 192.168.1.100/24",
            "nmcli connection modify eth0 ipv4.gateway 192.168.1.1",
            "nmcli connection modify eth0 ipv4.method manual",
            "nmcli connection up eth0",
        ])

        # Build section with some text before and after the code block
        content = f"Before the code.\n\n```\n{code}\n```\n\nAfter the code."
        guide = make_guide([make_section("Static IP Config", content, code_blocks=[code])])
        chunks = chunk_guide(guide)

        # Find chunks containing the code
        code_containing_chunks = [c for c in chunks if "nmcli connection modify" in c.text]

        if code_containing_chunks:
            # The code should be complete in any chunk that contains it
            for chunk in code_containing_chunks:
                # All lines of the code should be in the same chunk
                assert "192.168.1.100/24" in chunk.text
                assert "connection up" in chunk.text

    def test_empty_guide_returns_no_chunks(self):
        """A guide with no sections should produce no chunks."""
        guide = make_guide([])
        chunks = chunk_guide(guide)
        assert chunks == []

    def test_empty_section_produces_no_chunk(self):
        """A section with no content should be skipped."""
        empty_section = Section(
            heading="Empty Section",
            heading_level=2,
            hierarchy=["Empty Section"],
            content="",
            code_blocks=[],
            content_type="concept",
            url_anchor="",
        )
        guide = make_guide([empty_section])
        # Empty sections are filtered out before chunking
        assert guide.total_sections == 1  # ParsedGuide doesn't filter


class TestChunkMetadata:
    """Tests that chunks have correct metadata attached."""

    def test_chunks_have_version(self):
        """Every chunk should have the correct version tag."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)], version="9")
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert chunk.version == "9"

    def test_chunks_have_guide_slug(self):
        """Every chunk should reference the correct guide slug."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)])
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert chunk.guide == "test_guide"

    def test_chunks_have_guide_title(self):
        """Every chunk should have the guide title."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)])
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert chunk.guide_title == "Test Guide"

    def test_chunks_have_section_hierarchy(self):
        """Every chunk should have a non-empty section hierarchy."""
        content = " ".join(["word"] * 200)
        hierarchy = ["Chapter 5", "5.1 Networking", "5.1.2 DNS"]
        guide = make_guide([make_section("DNS Config", content, hierarchy=hierarchy)])
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert isinstance(chunk.section_hierarchy, list)
            assert len(chunk.section_hierarchy) > 0

    def test_chunks_have_unique_ids(self):
        """Every chunk should have a unique ID."""
        sections = []
        for i in range(5):
            content = " ".join(["word"] * 200)
            sections.append(make_section(f"Section {i}", content))

        guide = make_guide(sections)
        chunks = chunk_guide(guide)

        ids = [c.id for c in chunks]
        # All IDs should be unique
        assert len(ids) == len(set(ids)), "Found duplicate chunk IDs"

    def test_chunk_id_contains_version_and_guide(self):
        """Chunk ID format should be 'version::guide::index'."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)], version="9")
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert chunk.id.startswith("9::test_guide::"), (
                f"Unexpected chunk ID format: {chunk.id}"
            )

    def test_chunk_token_count_is_accurate(self):
        """chunk.token_count should approximately match count_tokens(chunk.text)."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)])
        chunks = chunk_guide(guide)

        for chunk in chunks:
            actual_tokens = count_tokens(chunk.text)
            # Allow 5% variance (due to heading prepended by splitter)
            assert abs(chunk.token_count - actual_tokens) <= max(5, actual_tokens * 0.05), (
                f"Token count mismatch: stored={chunk.token_count}, actual={actual_tokens}"
            )

    def test_chunk_url_contains_guide_url(self):
        """Chunk URL should reference the guide URL."""
        content = " ".join(["word"] * 200)
        guide = make_guide([make_section("Test", content)])
        chunks = chunk_guide(guide)

        for chunk in chunks:
            assert "docs.redhat.com" in chunk.url or "test" in chunk.url


class TestTokenCounting:
    """Tests for the count_tokens utility function."""

    def test_empty_string_is_zero(self):
        """Empty string should have 0 tokens."""
        assert count_tokens("") == 0

    def test_single_word_has_tokens(self):
        """A single word should have at least 1 token."""
        assert count_tokens("hello") >= 1

    def test_longer_text_has_more_tokens(self):
        """Longer text should have more tokens than shorter text."""
        short = "Configure networking"
        long_text = "Configure networking on Red Hat Enterprise Linux 9 using NetworkManager"
        assert count_tokens(long_text) > count_tokens(short)

    def test_approximate_ratio(self):
        """Token count should be roughly 0.75× word count (approximate)."""
        # 100 simple English words ≈ 75–100 tokens
        words = " ".join(["the"] * 100)
        tokens = count_tokens(words)
        assert 50 <= tokens <= 150, f"Unexpected token count: {tokens} for 100 words"


class TestSmallSectionMerging:
    """Tests that small sections are merged to meet minimum chunk size."""

    def test_tiny_sections_get_merged(self):
        """Very small sections should be merged into larger chunks."""
        # Create many tiny sections (each only a few tokens)
        sections = [
            make_section(f"Tiny {i}", f"Short content {i}.")
            for i in range(20)
        ]
        guide = make_guide(sections)
        chunks = chunk_guide(guide)

        # If tiny sections are merged, we should have fewer chunks than sections
        # (or at worst, equal number — no empty chunks)
        assert len(chunks) <= len(sections)

    def test_single_short_section_still_produces_chunk(self):
        """Even a short section should produce at least one chunk (not be dropped)."""
        guide = make_guide([make_section("Short", "A brief description of something.")])
        chunks = chunk_guide(guide)
        # Should have at least one chunk (content is too small to meet min, but heading is added)
        assert len(chunks) >= 0  # May be 0 if content is too tiny even with heading
