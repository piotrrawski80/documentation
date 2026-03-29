"""
chunker/splitter.py — Splits parsed section records into embedding-ready chunks.

Strategy (in priority order):
  1. Small/medium sections (≤ max_chars) → single chunk, no splitting.
  2. Large sections → split at semantic boundaries:
     a. code block fences (```...```)
     b. markdown table blocks (| ... |)
     c. double-newline paragraph breaks
     d. list item boundaries (lines starting with - or N.)
  3. Extremely large sections (package manifests, BPF feature lists) →
     aggressive paragraph-level splitting.

Invariants:
  - Code blocks are NEVER split across chunks.
  - Every chunk carries full parent hierarchy context.
  - chunk_id is deterministic: {parent_record_id}/c{chunk_index}

Usage:
    from rh_linux_docs_agent.chunker.splitter import chunk_section_record
    chunks = chunk_section_record(record)
"""

from __future__ import annotations

import hashlib
import re
import logging

import tiktoken

logger = logging.getLogger(__name__)

# ── Tokenizer (for accurate size estimation) ─────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 family tokenizer)."""
    return len(_enc.encode(text))


# ── Configuration ─────────────────────────────────────────────────────────────
MIN_TOKENS = 100     # Don't emit chunks smaller than this (merge with next)
TARGET_TOKENS = 500  # Ideal chunk size
MAX_TOKENS = 800     # Hard ceiling (except atomic code blocks)


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_guide(parsed_guide) -> list[dict]:
    """
    Chunk all sections of a ParsedGuide into embedding-ready dicts.

    Args:
        parsed_guide: A ParsedGuide instance (from parser.models).

    Returns:
        List of chunk dicts ready for embedding + storage.
    """
    all_chunks: list[dict] = []
    for record in parsed_guide.section_records():
        all_chunks.extend(chunk_section_record(record))
    return all_chunks


def chunk_section_record(record: dict) -> list[dict]:
    """
    Split a single parsed section record into one or more chunk dicts.

    Args:
        record: A flat section dict as produced by ParsedGuide.section_records().

    Returns:
        List of chunk dicts ready for JSON serialization.
        Single-chunk sections return a list with one element.
    """
    body = record.get("body_text", "")
    if not body.strip():
        return []

    # Split body into semantic segments
    segments = _segment_body(body)

    # Pack segments into chunks respecting token limits
    chunk_texts = _pack_segments(segments)

    # Build output records
    chunks: list[dict] = []
    parent_id = record["record_id"]
    heading = record["heading"]

    for idx, text in enumerate(chunk_texts):
        # Prepend heading to every chunk for retrieval context
        if heading and not text.startswith(heading):
            chunk_body = f"{heading}\n\n{text}"
        else:
            chunk_body = text

        chunk_id = f"{parent_id}/c{idx}"
        cc = len(chunk_body)
        wc = len(chunk_body.split())

        chunks.append({
            "chunk_id": chunk_id,
            "parent_record_id": parent_id,
            "chunk_index": idx,
            # ── inherited metadata (pass-through) ──
            "product": record["product"],
            "major_version": record["major_version"],
            "minor_version": record.get("minor_version"),
            "doc_type": record["doc_type"],
            "guide_slug": record["guide_slug"],
            "guide_title": record["guide_title"],
            "heading": heading,
            "hierarchy": record["hierarchy"],
            "heading_path_text": record["heading_path_text"],
            "section_id": record["section_id"],
            "guide_url": record["guide_url"],
            "section_url": record["section_url"],
            "content_type": record["content_type"],
            # ── chunk content ──
            "chunk_text": chunk_body,
            "char_count": cc,
            "word_count": wc,
            # ── helper flags (computed on this chunk, not inherited) ──
            "has_code_blocks": "```" in chunk_body,
            "has_tables": bool(re.search(r"\|\s*---", chunk_body)),
        })

    return chunks


# ── Segmentation ──────────────────────────────────────────────────────────────

# Regex that matches fenced code blocks (```...```) as atomic units
_CODE_BLOCK_RE = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)

# Regex that matches a contiguous markdown table (lines starting with |)
_TABLE_RE = re.compile(r"((?:^\|.+\|$\n?)+)", re.MULTILINE)


def _segment_body(body: str) -> list[str]:
    """
    Split body_text into semantic segments that can be recombined into chunks.

    Segment types (in parse order):
      1. Fenced code blocks → atomic, never split further.
      2. Markdown tables → atomic blocks.
      3. Everything else → split on double-newline (paragraph boundary).

    Returns a list of non-empty stripped strings.
    """
    segments: list[str] = []

    # Step 1: split out code blocks
    parts = _CODE_BLOCK_RE.split(body)

    for part in parts:
        if not part.strip():
            continue

        if part.startswith("```"):
            # Atomic code block
            segments.append(part.strip())
            continue

        # Step 2: within non-code text, split out tables
        table_parts = _TABLE_RE.split(part)
        for tp in table_parts:
            if not tp.strip():
                continue

            if tp.strip().startswith("|"):
                # Atomic table block
                segments.append(tp.strip())
                continue

            # Step 3: split remaining prose on paragraph boundaries
            paragraphs = re.split(r"\n\n+", tp)
            for para in paragraphs:
                para = para.strip()
                if para:
                    segments.append(para)

    return segments


# ── Packing ───────────────────────────────────────────────────────────────────

def _pack_segments(segments: list[str]) -> list[str]:
    """
    Greedily pack segments into chunks within token limits.

    Rules:
      - Accumulate segments until adding the next would exceed MAX_TOKENS.
      - When the buffer is full, yield it and start a new one.
      - Code blocks that exceed MAX_TOKENS → solo oversized chunk (never split).
      - Tables that exceed MAX_TOKENS → split into row-groups of ~TARGET size.
      - After the loop, any remaining buffer is yielded even if below MIN_TOKENS.
    """
    if not segments:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for seg in segments:
        seg_tokens = _tok(seg)

        # ── Oversized code block → solo chunk, never split ──
        if seg_tokens > MAX_TOKENS and seg.startswith("```"):
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            chunks.append(seg)
            continue

        # ── Oversized table → split into row-group sub-chunks ──
        if seg_tokens > MAX_TOKENS and seg.startswith("|"):
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            table_chunks = _split_large_table(seg)
            chunks.extend(table_chunks)
            continue

        # Would adding this segment overflow the buffer?
        projected = buf_tokens + seg_tokens + (1 if buf else 0)
        if projected > MAX_TOKENS and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [], 0

        # ── Oversized prose/list segment → split into sub-paragraphs ──
        if seg_tokens > MAX_TOKENS:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            sub_chunks = _split_large_prose(seg)
            chunks.extend(sub_chunks)
            continue

        buf.append(seg)
        buf_tokens += seg_tokens

    # Flush remainder
    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


def _split_large_table(table_text: str) -> list[str]:
    """
    Split an oversized markdown table into row-group chunks.

    Keeps the header (first 2 lines: column names + separator) at the
    top of every chunk so each chunk is a valid standalone table.
    """
    lines = table_text.split("\n")
    if len(lines) < 3:
        return [table_text]

    header = "\n".join(lines[:2])  # column names + |---|
    data_rows = lines[2:]

    chunks: list[str] = []
    buf_rows: list[str] = []
    buf_tokens = _tok(header)

    for row in data_rows:
        row_tokens = _tok(row)
        if buf_tokens + row_tokens > TARGET_TOKENS and buf_rows:
            chunk = header + "\n" + "\n".join(buf_rows)
            chunks.append(chunk)
            buf_rows = []
            buf_tokens = _tok(header)
        buf_rows.append(row)
        buf_tokens += row_tokens

    if buf_rows:
        chunk = header + "\n" + "\n".join(buf_rows)
        chunks.append(chunk)

    return chunks


def _split_large_prose(text: str) -> list[str]:
    """
    Split an oversized prose/list block into chunks at line boundaries.

    Splitting order of preference:
      1. Blank lines (paragraph breaks within the segment)
      2. List-item starts (lines beginning with '- ' or 'N. ')
      3. Any newline (last resort)

    Each resulting chunk targets TARGET_TOKENS size.
    """
    # Try paragraph-level splits first
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) > 1:
        return _pack_segments(paragraphs)

    # Single block — try splitting on list-item boundaries
    lines = text.split("\n")
    if len(lines) < 2:
        return [text]

    # Group lines into logical items (each starts with - or N.)
    items: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        is_item_start = (
            stripped.startswith("- ")
            or re.match(r"^\d+\.\s", stripped)
        )
        if is_item_start and current:
            items.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        items.append("\n".join(current))

    if len(items) <= 1:
        # No list structure found — hard split on lines
        items = lines

    # Pack items into chunks
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for item in items:
        item_tokens = _tok(item)
        if buf_tokens + item_tokens > TARGET_TOKENS and buf:
            chunks.append("\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(item)
        buf_tokens += item_tokens

    if buf:
        chunks.append("\n".join(buf))

    return chunks
