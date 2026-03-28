"""
chunker/models.py — Data classes for document chunks.

A chunk is the fundamental unit stored in the vector database. Each chunk is:
- Small enough to be precisely retrieved (150–800 tokens)
- Large enough to contain meaningful context (at least 150 tokens)
- Always associated with its section hierarchy (for context)
- Tagged with its RHEL version (for filtering)

Flow:
  ParsedGuide → Chunker → [Chunk, Chunk, ...] → Embedder → LanceDB
"""

import json
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage in the vector database.

    Each chunk represents a piece of documentation content from a specific
    section of a specific guide. The chunk is sized to be retrievable and
    contains enough context to be useful when passed to an LLM.

    Attributes:
        id: Unique identifier for this chunk. Format:
            "{version}::{guide_slug}::{section_index}::{chunk_index}"
            Example: "9::configuring_networking::12::0"
        version: RHEL major version, e.g. "9". Used for filtering.
        guide: URL slug of the source guide, e.g. "configuring_networking".
        guide_title: Human-readable guide title for display in search results.
        section_hierarchy: List of heading texts from root to current section.
                           Example: ["Chapter 5. Networking", "5.1 NetworkManager"]
                           This gives context about where in the guide this chunk is.
        content_type: "procedure", "concept", or "reference".
        text: The actual chunk text. This is what gets embedded and what the LLM reads.
        token_count: Number of tokens in the text. Used for monitoring chunk sizes.
        url: Full URL to the source section on docs.redhat.com.
    """

    id: str
    version: str
    guide: str
    guide_title: str
    section_hierarchy: list[str]
    content_type: str
    text: str
    token_count: int
    url: str

    def section_hierarchy_json(self) -> str:
        """
        Return section_hierarchy serialized as a JSON string.

        LanceDB stores this as a string column. JSON is used because it's
        human-readable and easy to parse back into a Python list.
        """
        return json.dumps(self.section_hierarchy)

    @property
    def section_path(self) -> str:
        """
        Return a human-readable path string like "Chapter 5 > 5.1 Networking".
        Useful for displaying in search result summaries.
        """
        return " > ".join(self.section_hierarchy)
