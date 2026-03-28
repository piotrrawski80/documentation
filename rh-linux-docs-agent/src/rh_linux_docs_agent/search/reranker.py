"""
search/reranker.py — Cross-encoder reranking for retrieved chunks.

After initial retrieval (vector + BM25), a cross-encoder model scores each
(query, chunk) pair directly. This produces much more accurate relevance
scores than bi-encoder similarity, at the cost of being slower (O(n) passes
through the model, where n = number of candidates).

Pipeline position:
  query → HybridSearch.search(top_k=20) → Reranker.rerank(top_n=5) → [final results]

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80MB, runs on CPU
  - Trained on MS MARCO passage ranking
  - Good quality on technical English

Usage:
    reranker = Reranker()
    reranked = reranker.rerank(query="configure static IP", candidates=results, top_n=5)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.

    Lazily loads the model on first call to avoid importing torch at import time.

    Attributes:
        model_name: HuggingFace model name for the cross-encoder.
        _model: The loaded CrossEncoder instance (None until first use).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Load the cross-encoder model. Called once on first rerank()."""
        if self._model is not None:
            return
        logger.info(f"Loading reranker model: {self.model_name}")
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate chunks using cross-encoder scoring.

        Args:
            query:      The original search query.
            candidates: List of result dicts from HybridSearch.search().
            top_n:      How many top results to return after reranking.
            text_key:   Dict key containing the chunk text to score against.

        Returns:
            Top-n results sorted by cross-encoder score (best first).
            Each result dict gets a '_rerank_score' key (float, higher = better)
            and keeps its original '_score' from retrieval.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            # No reranking needed for 0-1 results
            for c in candidates:
                c["_rerank_score"] = c.get("_score", 0.0)
            return candidates

        self._load_model()

        # Build (query, passage) pairs for cross-encoder scoring
        pairs = []
        for c in candidates:
            text = c.get(text_key, "")
            # Truncate very long texts — cross-encoder max is ~512 tokens
            if len(text) > 2000:
                text = text[:2000]
            pairs.append((query, text))

        # Score all pairs in one batch
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored = []
        for i, c in enumerate(candidates):
            result = dict(c)
            result["_rerank_score"] = float(scores[i])
            scored.append(result)

        scored.sort(key=lambda r: r["_rerank_score"], reverse=True)

        return scored[:top_n]


class NoOpReranker:
    """
    Pass-through reranker that does no reranking.

    Use this when you want to skip the reranking step (e.g., for speed
    comparison or when the cross-encoder model is unavailable).
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict[str, Any]]:
        """Return top_n candidates without reranking."""
        for c in candidates:
            c["_rerank_score"] = c.get("_score", 0.0)
        return candidates[:top_n]
