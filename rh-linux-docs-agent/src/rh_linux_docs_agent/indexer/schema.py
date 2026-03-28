"""
indexer/schema.py — PyArrow schema for the LanceDB chunks table.

Defines all columns stored per chunk, matching the output of the chunking
pipeline (data/chunked/rhel9/*.chunks.json) plus the embedding vector.

The schema must stay in sync with:
  - chunker/splitter.py  (produces the chunk dicts)
  - store.py             (inserts/queries using this schema)

Column naming convention: matches chunk JSON keys exactly, except:
  - 'vector' is the embedding column (not in JSON, added at index time)
  - 'hierarchy' is stored as a JSON string (Arrow doesn't support variable-length lists well)
"""

import json
import pyarrow as pa

from rh_linux_docs_agent.config import settings


def get_schema() -> pa.Schema:
    """
    Return the Apache Arrow schema for the LanceDB chunks table.

    The 'vector' column is a fixed-size list of float32 — this is what
    LanceDB uses for approximate nearest-neighbor (ANN) search.

    Returns:
        PyArrow Schema object defining all table columns.
    """
    return pa.schema([
        # ── identifiers ──
        pa.field("chunk_id", pa.string()),
        pa.field("parent_record_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        # ── product metadata ──
        pa.field("product", pa.string()),
        pa.field("major_version", pa.string()),
        pa.field("minor_version", pa.string()),          # nullable
        pa.field("doc_type", pa.string()),
        # ── guide metadata ──
        pa.field("guide_slug", pa.string()),
        pa.field("guide_title", pa.string()),
        pa.field("guide_url", pa.string()),
        pa.field("section_url", pa.string()),
        # ── section structure ──
        pa.field("heading", pa.string()),
        pa.field("hierarchy", pa.string()),               # JSON-encoded list
        pa.field("heading_path_text", pa.string()),
        pa.field("section_id", pa.string()),
        pa.field("content_type", pa.string()),
        # ── chunk content ──
        pa.field("chunk_text", pa.string()),
        pa.field("char_count", pa.int32()),
        pa.field("word_count", pa.int32()),
        # ── flags ──
        pa.field("has_code_blocks", pa.bool_()),
        pa.field("has_tables", pa.bool_()),
        # ── embedding vector ──
        pa.field("vector", pa.list_(pa.float32(), settings.embedding_dim)),
    ])


def chunk_dict_to_record(chunk: dict, vector: list[float]) -> dict:
    """
    Convert a chunk dict (from chunks.json) + embedding vector into a
    LanceDB-ready record whose keys match get_schema().

    The 'hierarchy' list is serialized to JSON for storage.
    None values for minor_version are stored as empty string.
    """
    return {
        "chunk_id": chunk["chunk_id"],
        "parent_record_id": chunk["parent_record_id"],
        "chunk_index": chunk["chunk_index"],
        "product": chunk["product"],
        "major_version": chunk["major_version"],
        "minor_version": chunk.get("minor_version") or "",
        "doc_type": chunk["doc_type"],
        "guide_slug": chunk["guide_slug"],
        "guide_title": chunk["guide_title"],
        "guide_url": chunk["guide_url"],
        "section_url": chunk["section_url"],
        "heading": chunk["heading"],
        "hierarchy": json.dumps(chunk["hierarchy"], ensure_ascii=False),
        "heading_path_text": chunk["heading_path_text"],
        "section_id": chunk["section_id"],
        "content_type": chunk["content_type"],
        "chunk_text": chunk["chunk_text"],
        "char_count": chunk["char_count"],
        "word_count": chunk["word_count"],
        "has_code_blocks": chunk["has_code_blocks"],
        "has_tables": chunk["has_tables"],
        "vector": vector,
    }


def record_to_result(record: dict) -> dict:
    """
    Convert a LanceDB query result back into a rich dict for display/search.

    Parses the JSON-encoded hierarchy back into a Python list and removes
    the raw vector (too large to display).
    """
    result = dict(record)

    # Parse hierarchy JSON string → list
    if isinstance(result.get("hierarchy"), str):
        try:
            result["hierarchy"] = json.loads(result["hierarchy"])
        except (json.JSONDecodeError, TypeError):
            result["hierarchy"] = []

    # Drop raw vector from display results
    result.pop("vector", None)

    return result
