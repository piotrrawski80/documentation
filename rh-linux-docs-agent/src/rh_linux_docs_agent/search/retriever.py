"""
search/retriever.py — Main retrieval pipeline: retrieve → rerank → filter → return.

This is the single entry point for all retrieval operations.
It composes HybridSearch + Reranker + post-processing (score threshold,
deduplication, query classification) into a clean pipeline.

Pipeline:
  query → classify → HybridSearch(top_k) → Reranker(top_n)
        → score threshold → dedup → results

Usage:
    retriever = Retriever()
    results = retriever.retrieve("configure static IP", major_version="9")

    # Without reranking (faster)
    retriever = Retriever(use_reranker=False)

    # Vector-only mode
    results = retriever.retrieve("...", mode="vector")
"""

import logging
import re
from typing import Any

from rh_linux_docs_agent.search.hybrid import HybridSearch
from rh_linux_docs_agent.search.reranker import Reranker, NoOpReranker
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)


# ── Query classification ────────────────────────────────────────────────────

# Simple keyword-based classifier.  Returns one of:
#   "procedure"        – how-to, step-by-step
#   "troubleshooting"  – debug, fix, error
#   "concept"          – what is, explain, overview
#   "reference"        – list, show, default value

_QUERY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("procedure", re.compile(
        r"\b(how\s+(?:to|do|can)|configure|set\s*up|install|create|enable|"
        r"disable|add|remove|join|mount|build|deploy|run|start|stop|restart|"
        r"steps?\s+(?:to|for))\b",
        re.IGNORECASE,
    )),
    ("troubleshooting", re.compile(
        r"\b(troubleshoot|debug|diagnos|fix|error|fail|denied|not\s+working|"
        r"issue|problem|cannot|can't|won't|unable|resolve|audit2allow|"
        r"selinux\s+denial|permission\s+denied)\b",
        re.IGNORECASE,
    )),
    ("reference", re.compile(
        r"\b(list|show|default|values?|options?|parameters?|available|"
        r"supported|what\s+(?:are|is)\s+the\s+(?:default|available)|"
        r"release\s+notes?|changes?\s+in|changelog)\b",
        re.IGNORECASE,
    )),
    ("concept", re.compile(
        r"\b(what\s+is|explain|overview|difference\s+between|concept|"
        r"purpose\s+of|why\s+(?:does|is|do)|architecture|how\s+does)\b",
        re.IGNORECASE,
    )),
]

# Map query type → preferred content_type values (used as a soft boost, not
# a hard filter — we only apply it when the caller hasn't set content_type).
_QUERY_TYPE_CONTENT_BIAS: dict[str, list[str]] = {
    "procedure":       ["procedure", "assembly"],
    "troubleshooting": ["procedure", "reference"],
    "concept":         ["concept", "assembly"],
    "reference":       ["reference"],
}


def classify_query(query: str) -> str:
    """
    Classify a query into one of: procedure, troubleshooting, concept, reference.

    Uses keyword patterns. Returns "procedure" as default when no pattern matches
    (most RHEL documentation questions are how-to).

    >>> classify_query("How to configure firewalld?")
    'procedure'
    >>> classify_query("Troubleshoot SELinux denials")
    'troubleshooting'
    >>> classify_query("What is the default crypto policy?")
    'concept'
    """
    for label, pattern in _QUERY_PATTERNS:
        if pattern.search(query):
            return label
    return "procedure"


# ── Deduplication ────────────────────────────────────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """
    Compute a cheap token-overlap ratio between two strings.

    Uses word-level Jaccard similarity:  |A ∩ B| / |A ∪ B|
    Returns a float in [0, 1].
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _deduplicate(
    results: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Remove near-duplicate chunks from results.

    A chunk is considered a duplicate if:
      (a) it shares the exact same heading as a higher-scored chunk, OR
      (b) its chunk_text has Jaccard word-overlap ≥ threshold with a
          higher-scored chunk.

    Results must already be sorted best-first (they are after reranking).
    The first (highest-scoring) occurrence is always kept.
    """
    if threshold >= 1.0:
        return results          # dedup disabled

    kept: list[dict[str, Any]] = []

    for r in results:
        heading = r.get("heading", "")
        text = r.get("chunk_text", "")
        is_dup = False

        for k in kept:
            # Same heading in same guide → almost certainly duplicate
            if (
                heading
                and heading == k.get("heading", "")
                and r.get("guide_slug", "") == k.get("guide_slug", "")
            ):
                is_dup = True
                break

            # High text overlap
            if _text_overlap(text, k.get("chunk_text", "")) >= threshold:
                is_dup = True
                break

        if is_dup:
            logger.debug(
                "Dedup: dropping chunk '%s' (guide=%s) as duplicate",
                heading[:60], r.get("guide_slug", ""),
            )
        else:
            kept.append(r)

    return kept


# ── Retriever ────────────────────────────────────────────────────────────────

class Retriever:
    """
    Full retrieval pipeline: classify → hybrid search → rerank → threshold → dedup.

    Args:
        use_reranker: If True (default), load and use cross-encoder reranker.
                      If False, skip reranking (faster, slightly lower quality).
        store:        Optional DocStore to reuse.
        embedder:     Optional Embedder to reuse.
    """

    def __init__(
        self,
        use_reranker: bool = True,
        store=None,
        embedder=None,
    ) -> None:
        self.searcher = HybridSearch(store=store, embedder=embedder)
        self.reranker = Reranker() if use_reranker else NoOpReranker()
        self._use_reranker = use_reranker

    def retrieve(
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
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Full retrieval pipeline: classify → search → rerank → filter → return.

        Args:
            query:          Search query.
            top_k:          Candidates to retrieve (default: settings.search_top_k).
            top_n:          Final results after reranking (default: settings.search_rerank_top_n).
            product:        Filter by product.
            major_version:  Filter by RHEL version.
            minor_version:  Filter by minor version.
            doc_type:       Filter by doc_type.
            guide_slug:     Filter by guide.
            content_type:   Filter by content type.
            mode:           "hybrid", "vector", or "bm25".

        Returns:
            List of result dicts sorted by final relevance score.
            Each result has '_score', '_rerank_score', and '_query_type'.
        """
        k = top_k or settings.search_top_k
        n = top_n or settings.search_rerank_top_n

        # ── Step 0: Classify query ──────────────────────────────────────
        query_type = classify_query(query)
        logger.debug("Query classified as '%s': %s", query_type, query[:60])

        # ── Step 1: Retrieve candidates ─────────────────────────────────
        # Fetch extra candidates (k) to give reranker + dedup room.
        candidates = self.searcher.search(
            query,
            top_k=k,
            top_n=k,  # Get all k candidates for reranking
            product=product,
            major_version=major_version,
            minor_version=minor_version,
            doc_type=doc_type,
            guide_slug=guide_slug,
            content_type=content_type,
            mode=mode,
        )

        if not candidates:
            return []

        # ── Step 2: Rerank ──────────────────────────────────────────────
        # Ask reranker for slightly more than final n so dedup still has room.
        rerank_n = min(n + 3, len(candidates))
        results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=rerank_n,
        )

        # ── Step 3: Score threshold (only when cross-encoder is active) ─
        if self._use_reranker:
            threshold = settings.rerank_score_threshold
            before = len(results)
            results = [
                r for r in results
                if r.get("_rerank_score", 0.0) >= threshold
            ]
            if len(results) < before:
                logger.debug(
                    "Score threshold %.1f removed %d/%d chunks",
                    threshold, before - len(results), before,
                )

        # ── Step 4: Deduplicate ─────────────────────────────────────────
        results = _deduplicate(results, settings.dedup_similarity_threshold)

        # ── Step 5: Trim to final n ─────────────────────────────────────
        results = results[:n]

        # ── Step 6: Attach query classification ─────────────────────────
        for r in results:
            r["_query_type"] = query_type

        logger.debug(
            "Retriever '%s': %d candidates → %d final "
            "(reranker=%s, type=%s)",
            query[:40], len(candidates), len(results),
            "on" if self._use_reranker else "off", query_type,
        )

        return results
