"""
indexer/store.py — LanceDB vector database operations for RHEL doc chunks.

LanceDB is an embedded vector database — data stored as files in a directory
(data/lancedb/) without requiring a running database server.

Key operations:
  - create_or_open_table(): Get/create the "rhel_docs" table
  - insert_chunks(): Batch-insert chunk dicts + vectors with deduplication
  - delete_by_guide(): Remove all chunks for a specific guide
  - delete_by_version(): Remove all chunks for a RHEL version
  - create_indexes(): Build IVF-PQ vector + FTS/BM25 indexes
  - search_vector(): Pure vector similarity search
  - table_stats(): Summary of what's in the database

Usage:
    store = DocStore()
    store.insert_chunks(chunks, vectors)
    results = store.search_vector(query_vector, limit=5)
"""

import logging
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from rh_linux_docs_agent.indexer.schema import get_schema, chunk_dict_to_record, record_to_result
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)


class DocStore:
    """
    Manages the LanceDB vector database for RHEL documentation chunks.

    All chunks across all RHEL versions live in a single table.
    Version/guide-specific queries use column filtering.

    Database location: settings.db_path  (default: data/lancedb/)
    Table name:        settings.table_name (default: "rhel_docs")
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Connect to LanceDB, creating the directory if needed."""
        path = Path(db_path) if db_path else Path(settings.db_path)
        path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self._db_path = path
        logger.debug(f"Connected to LanceDB at {path}")

    def create_or_open_table(self) -> Any:
        """Get the chunks table, creating it if it doesn't exist."""
        table_names = self.db.table_names()

        if settings.table_name not in table_names:
            logger.info(f"Creating new LanceDB table '{settings.table_name}'")
            schema = get_schema()
            empty_table = pa.table(
                {field.name: [] for field in schema},
                schema=schema,
            )
            table = self.db.create_table(
                settings.table_name,
                data=empty_table,
                mode="create",
            )
        else:
            table = self.db.open_table(settings.table_name)
            logger.debug(
                f"Opened existing table '{settings.table_name}' "
                f"with {table.count_rows()} rows"
            )
        return table

    def insert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert chunk dicts + embedding vectors into the database.

        Deduplicates by deleting existing records with matching chunk_ids
        before inserting (idempotent upsert pattern).

        Args:
            chunks:     List of chunk dicts (from chunks.json).
            vectors:    List of embedding vectors, same order as chunks.
            batch_size: Rows per database write.

        Returns:
            Number of chunks inserted.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return 0

        table = self.create_or_open_table()

        # Delete existing records with these IDs (upsert pattern)
        chunk_ids = [c["chunk_id"] for c in chunks]
        if table.count_rows() > 0:
            for i in range(0, len(chunk_ids), 500):
                batch_ids = chunk_ids[i : i + 500]
                id_list = ", ".join(f"'{cid}'" for cid in batch_ids)
                try:
                    table.delete(f"chunk_id IN ({id_list})")
                except Exception as e:
                    logger.warning(f"Delete error (non-fatal): {e}")

        # Convert and insert in batches
        total_inserted = 0
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_vectors = vectors[batch_start : batch_start + batch_size]

            records = [
                chunk_dict_to_record(chunk, vector)
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]
            table.add(records)
            total_inserted += len(records)
            logger.debug(
                f"Inserted batch: {len(records)} chunks "
                f"(total so far: {total_inserted})"
            )

        return total_inserted

    def search_vector(
        self,
        query_vector: list[float],
        limit: int = 10,
        version_filter: str | None = None,
        doc_type_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for similar chunks using vector cosine similarity.

        Args:
            query_vector:    384-dim float list from Embedder.embed_query().
            limit:           Max results to return.
            version_filter:  Optional RHEL version filter (e.g. "9").
            doc_type_filter: Optional doc_type filter (e.g. "networking").

        Returns:
            List of result dicts sorted by relevance, with '_distance' score.
        """
        table = self.create_or_open_table()

        query = table.search(query_vector).limit(limit)

        # Apply optional filters
        where_clauses = []
        if version_filter:
            where_clauses.append(f"major_version = '{version_filter}'")
        if doc_type_filter:
            where_clauses.append(f"doc_type = '{doc_type_filter}'")
        if where_clauses:
            query = query.where(" AND ".join(where_clauses))

        results = query.to_list()
        return [record_to_result(r) for r in results]

    def delete_by_version(self, version: str) -> int:
        """Delete all chunks for a specific RHEL version."""
        table = self.create_or_open_table()
        before = table.count_rows()
        if before == 0:
            return 0
        table.delete(f"major_version = '{version}'")
        deleted = before - table.count_rows()
        logger.info(f"Deleted {deleted} chunks for RHEL {version}")
        return deleted

    def delete_by_guide(self, guide_slug: str) -> int:
        """Delete all chunks for a specific guide."""
        table = self.create_or_open_table()
        before = table.count_rows()
        if before == 0:
            return 0
        table.delete(f"guide_slug = '{guide_slug}'")
        deleted = before - table.count_rows()
        logger.info(f"Deleted {deleted} chunks for guide '{guide_slug}'")
        return deleted

    def create_indexes(self) -> None:
        """
        Build vector (IVF-PQ) and full-text (BM25) search indexes.

        Call after inserting all chunks. Vector index makes ANN search fast;
        FTS index enables BM25 keyword search via Tantivy.
        """
        table = self.create_or_open_table()
        row_count = table.count_rows()

        if row_count == 0:
            logger.warning("Table is empty — skipping index creation")
            return

        logger.info(f"Building search indexes for {row_count:,} rows...")

        # Vector index (IVF-PQ)
        # Partition count scaled to data size; sqrt(n) is a good heuristic
        import math
        num_partitions = min(256, max(4, int(math.sqrt(row_count))))
        num_sub_vectors = min(96, settings.embedding_dim)

        try:
            logger.info(
                f"Building vector index (IVF-PQ, {num_partitions} partitions)..."
            )
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                vector_column_name="vector",
                replace=True,
            )
            logger.info("Vector index built successfully")
        except Exception as e:
            logger.warning(f"Vector index creation failed (will use brute-force): {e}")

        # Full-text search index (Tantivy/BM25)
        try:
            logger.info("Building full-text search (BM25) index on chunk_text...")
            table.create_fts_index("chunk_text", replace=True)
            logger.info("FTS index built successfully")
        except Exception as e:
            logger.warning(f"FTS index creation failed: {e}")

    def table_stats(self) -> dict:
        """Return summary statistics about the current table."""
        if settings.table_name not in self.db.table_names():
            return {"total_rows": 0}

        table = self.create_or_open_table()
        total = table.count_rows()
        if total == 0:
            return {"total_rows": 0}

        try:
            df = table.to_pandas(
                columns=["major_version", "doc_type", "guide_slug"],
                # LanceDB may need explicit vector exclusion
            )
            versions = df["major_version"].value_counts().to_dict()
            doc_types = df["doc_type"].value_counts().to_dict()
            guides = int(df["guide_slug"].nunique())
        except Exception:
            # Fallback: scan a sample
            try:
                rows = table.search().limit(10000).select(
                    ["major_version", "doc_type", "guide_slug"]
                ).to_list()
                from collections import Counter
                versions = dict(Counter(r["major_version"] for r in rows))
                doc_types = dict(Counter(r["doc_type"] for r in rows))
                guides = len(set(r["guide_slug"] for r in rows))
            except Exception:
                versions, doc_types, guides = {}, {}, 0

        return {
            "total_rows": total,
            "versions": versions,
            "doc_types": doc_types,
            "unique_guides": guides,
        }

    def get_total_count(self) -> int:
        """Total rows in the table, or 0 if table doesn't exist."""
        if settings.table_name not in self.db.table_names():
            return 0
        return self.create_or_open_table().count_rows()

    def list_versions(self) -> dict[str, int]:
        """Return a dict of {version: chunk_count} for all versions in this DB."""
        if settings.table_name not in self.db.table_names():
            return {}
        stats = self.table_stats()
        return stats.get("versions", {})

    def drop_table(self) -> None:
        """Drop the entire table. Use with caution."""
        if settings.table_name in self.db.table_names():
            self.db.drop_table(settings.table_name)
            logger.info(f"Dropped table '{settings.table_name}'")
