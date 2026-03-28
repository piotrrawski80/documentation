"""
search/hybrid.py — Hybrid search: vector + BM25 + RRF fusion.

Combines two retrieval methods for better recall:
  - Vector (semantic): finds conceptually similar content even without keyword overlap
  - BM25 (keyword): finds exact matches for commands, paths, package names, errors
  - RRF fusion: merges both ranked lists using Reciprocal Rank Fusion

Supports metadata filtering on all indexed columns:
  product, major_version, minor_version, doc_type, guide_slug, content_type

Pipeline position:
  query → HybridSearch.search() → [candidates] → Reranker → [top results]

Usage:
    searcher = HybridSearch()
    results = searcher.search("configure static IP", version="9", top_k=20)
"""

import logging
from typing import Any

from rh_linux_docs_agent.config import settings
from rh_linux_docs_agent.indexer.embedder import Embedder
from rh_linux_docs_agent.indexer.store import DocStore
from rh_linux_docs_agent.indexer.schema import record_to_result

logger = logging.getLogger(__name__)

# Columns to return from search (everything except the raw vector)
_SELECT_COLS = [
    "chunk_id", "parent_record_id", "chunk_index",
    "product", "major_version", "minor_version", "doc_type",
    "guide_slug", "guide_title", "guide_url", "section_url",
    "heading", "hierarchy", "heading_path_text", "section_id",
    "content_type", "chunk_text", "char_count", "word_count",
    "has_code_blocks", "has_tables",
]


def _build_where_clause(
    *,
    product: str | None = None,
    major_version: str | None = None,
    minor_version: str | None = None,
    doc_type: str | None = None,
    guide_slug: str | None = None,
    content_type: str | None = None,
) -> str | None:
    """Build a SQL WHERE clause from optional metadata filters."""
    clauses: list[str] = []
    if product:
        clauses.append(f"product = '{product}'")
    if major_version:
        clauses.append(f"major_version = '{major_version}'")
    if minor_version:
        clauses.append(f"minor_version = '{minor_version}'")
    if doc_type:
        clauses.append(f"doc_type = '{doc_type}'")
    if guide_slug:
        clauses.append(f"guide_slug = '{guide_slug}'")
    if content_type:
        clauses.append(f"content_type = '{content_type}'")
    return " AND ".join(clauses) if clauses else None


class HybridSearch:
    """
    Hybrid retrieval engine: vector similarity + BM25 keyword search + RRF fusion.

    Supports full metadata filtering on product, major_version, minor_version,
    doc_type, guide_slug, content_type.

    Example:
        searcher = HybridSearch()
        results = searcher.search("configure static IP", major_version="9")
    """

    def __init__(self, store: DocStore | None = None, embedder: Embedder | None = None) -> None:
        self.store = store or DocStore()
        self.embedder = embedder or Embedder()
        logger.debug("HybridSearch initialized")

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        # ── metadata filters ──
        product: str | None = None,
        major_version: str | None = None,
        minor_version: str | None = None,
        doc_type: str | None = None,
        guide_slug: str | None = None,
        content_type: str | None = None,
        # ── search mode ──
        mode: str = "hybrid",  # "hybrid", "vector", "bm25"
    ) -> list[dict[str, Any]]:
        """
        Search for relevant documentation chunks.

        Args:
            query:          Natural language or keyword query.
            top_k:          Candidates per method before fusion (default: settings.search_top_k).
            top_n:          Final results after fusion (default: settings.search_rerank_top_n).
            product:        Filter by product (e.g. "rhel").
            major_version:  Filter by major version (e.g. "9").
            minor_version:  Filter by minor version (e.g. "9.4").
            doc_type:       Filter by doc_type (e.g. "networking", "security").
            guide_slug:     Filter by specific guide.
            content_type:   Filter by content_type ("procedure", "concept", "reference").
            mode:           "hybrid" (default), "vector", or "bm25".

        Returns:
            List of result dicts with '_score' (RRF) or '_distance' (vector-only),
            sorted by relevance (best first).
        """
        k = top_k or settings.search_top_k
        n = top_n or settings.search_rerank_top_n

        if self.store.get_total_count() == 0:
            logger.warning("Database is empty — run indexing first")
            return []

        table = self.store.create_or_open_table()

        where = _build_where_clause(
            product=product,
            major_version=major_version,
            minor_version=minor_version,
            doc_type=doc_type,
            guide_slug=guide_slug,
            content_type=content_type,
        )

        vector_results: list[dict] = []
        bm25_results: list[dict] = []

        # ── Vector search ─────────────────────────────────────────────────
        if mode in ("hybrid", "vector"):
            query_vector = self.embedder.embed_query(query)
            vector_results = _run_vector_search(table, query_vector, where, k)

        # ── BM25 keyword search ───────────────────────────────────────────
        if mode in ("hybrid", "bm25"):
            bm25_results = _run_bm25_search(table, query, where, k)

        # ── Fusion / selection ────────────────────────────────────────────
        if mode == "hybrid" and vector_results and bm25_results:
            fused = _reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=settings.rrf_k,
                vector_weight=settings.vector_weight,
            )
        elif mode == "hybrid" and vector_results:
            # BM25 failed or returned nothing — fall back to vector-only
            fused = _add_scores(vector_results, source="vector")
        elif mode == "hybrid" and bm25_results:
            fused = _add_scores(bm25_results, source="bm25")
        elif mode == "vector":
            fused = _add_scores(vector_results, source="vector")
        elif mode == "bm25":
            fused = _add_scores(bm25_results, source="bm25")
        else:
            fused = []

        results = fused[:n]
        enriched = [record_to_result(r) for r in results]

        logger.debug(
            f"Search '{query[:50]}' (mode={mode}, where={where}): "
            f"{len(vector_results)} vec + {len(bm25_results)} bm25 → "
            f"{len(enriched)} results"
        )
        return enriched

    def search_vector_only(
        self, query: str, *, major_version: str | None = None, top_n: int = 10, **filters
    ) -> list[dict]:
        """Convenience: vector-only search."""
        return self.search(query, mode="vector", major_version=major_version, top_n=top_n, **filters)

    def search_bm25_only(
        self, query: str, *, major_version: str | None = None, top_n: int = 10, **filters
    ) -> list[dict]:
        """Convenience: BM25-only search."""
        return self.search(query, mode="bm25", major_version=major_version, top_n=top_n, **filters)


# ── Internal search functions ─────────────────────────────────────────────────

def _run_vector_search(
    table: Any, query_vector: list[float], where: str | None, limit: int
) -> list[dict]:
    """Cosine similarity search on embedding vectors."""
    try:
        q = (
            table.search(query_vector, vector_column_name="vector")
            .metric("cosine")
            .limit(limit)
            .select(_SELECT_COLS)
        )
        if where:
            q = q.where(where, prefilter=True)
        return q.to_list()
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


def _run_bm25_search(
    table: Any, query: str, where: str | None, limit: int
) -> list[dict]:
    """BM25 full-text keyword search (requires FTS index on chunk_text)."""
    try:
        q = (
            table.search(query, query_type="fts")
            .limit(limit)
            .select(_SELECT_COLS)
        )
        if where:
            q = q.where(where, prefilter=True)
        return q.to_list()
    except Exception as e:
        logger.warning(f"BM25 search failed (FTS index may not exist): {e}")
        return []


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    score(d) = vec_weight/(k + rank_vec) + bm25_weight/(k + rank_bm25)

    Results appearing in both lists get combined scores.
    """
    bm25_weight = 1.0 - vector_weight
    scores: dict[str, float] = {}
    result_by_id: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, start=1):
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + vector_weight / (k + rank)
        result_by_id[cid] = r

    for rank, r in enumerate(bm25_results, start=1):
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank)
        if cid not in result_by_id:
            result_by_id[cid] = r

    sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)

    fused: list[dict] = []
    for cid in sorted_ids:
        result = dict(result_by_id[cid])
        result["_score"] = round(scores[cid], 6)
        result["_search_mode"] = "hybrid"
        fused.append(result)

    return fused


def _add_scores(results: list[dict], source: str) -> list[dict]:
    """Add _score and _search_mode to single-source results."""
    scored = []
    for rank, r in enumerate(results, start=1):
        result = dict(r)
        # Normalize distance to score (1.0 = perfect match)
        dist = r.get("_distance", 0)
        result["_score"] = round(1.0 - dist, 6) if dist else round(1.0 / rank, 6)
        result["_search_mode"] = source
        scored.append(result)
    return scored
