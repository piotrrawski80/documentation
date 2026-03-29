"""
search/retriever.py — Main retrieval pipeline: retrieve → rerank → filter → return.

This is the single entry point for all retrieval operations.
It composes HybridSearch + Reranker + post-processing (score threshold,
deduplication, query classification, interface intent reranking) into a
clean pipeline.

Pipeline:
  query → classify → detect interface intent → HybridSearch(top_k)
        → Reranker(top_n) → score threshold → interface boost/penalty
        → dedup → mismatch check → results

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


# ── Interface intent detection ───────────────────────────────────────────────

# Detect whether the user is asking for CLI / terminal instructions or
# GUI / web-console instructions.  Returns "cli", "gui", or "neutral".

_CLI_INTENT_RE = re.compile(
    r"\b(command\s*line|cli\b|terminal|shell|bash|nmcli|firewall-cmd|"
    r"lvextend|lvcreate|vgcreate|pvcreate|lvresize|dnf\b|rpm\b|systemctl|"
    r"sysctl|nmstatectl|podman\b|semanage|restorecon|setsebool)\b",
    re.IGNORECASE,
)

_GUI_INTENT_RE = re.compile(
    r"\b(web\s*console|cockpit|gui\b|graphical|graphic\s+interface|"
    r"browser\s+interface|installer|anaconda|rhel\s+web\s+console)\b",
    re.IGNORECASE,
)


def detect_interface_intent(query: str) -> str:
    """
    Detect whether a query asks for CLI or GUI instructions.

    Returns:
        "cli"     — user explicitly wants command-line/terminal procedures
        "gui"     — user explicitly wants web-console/graphical procedures
        "neutral" — no explicit preference

    >>> detect_interface_intent("configure LVM from the command line")
    'cli'
    >>> detect_interface_intent("configure firewalld using the web console")
    'gui'
    >>> detect_interface_intent("configure LVM on RHEL 9")
    'neutral'
    """
    has_cli = bool(_CLI_INTENT_RE.search(query))
    has_gui = bool(_GUI_INTENT_RE.search(query))

    if has_cli and not has_gui:
        return "cli"
    if has_gui and not has_cli:
        return "gui"
    # Both or neither → neutral (don't bias)
    return "neutral"


# ── Chunk interface classification ───────────────────────────────────────────

# Heuristics applied to chunk text + heading to label each chunk as
# "cli", "gui", or "neutral".

_CHUNK_GUI_RE = re.compile(
    r"\b(web\s*console|cockpit|rhel\s+web\s+console|graphical\s+interface|"
    r"browser\b.*\binterface|anaconda\s+installer|gnome\s+settings|"
    r"click\s+(?:the|on|add|edit|save|apply|enable|remove)|"
    r"select\s+(?:the|from)|in\s+the\s+navigation)\b",
    re.IGNORECASE,
)

_CHUNK_CLI_RE = re.compile(
    r"(?:"
    # fenced code blocks with shell prompts
    r"^(?:\$|#)\s+\S"
    r"|```(?:bash|shell|console)?"
    # common RHEL CLI tools
    r"|\b(?:nmcli|firewall-cmd|lvextend|lvcreate|vgcreate|pvcreate|lvresize|"
    r"dnf\s+install|rpm\s+-|systemctl\s+|sysctl\s+|semanage\s+|restorecon\s+|"
    r"podman\s+|buildah\s+|subscription-manager|nmstatectl|timedatectl|"
    r"hostnamectl|localectl|journalctl|ausearch|audit2allow)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _classify_chunk_interface(chunk: dict[str, Any]) -> str:
    """
    Label a single chunk as "cli", "gui", or "neutral" based on its content.

    Checks heading + chunk_text for GUI and CLI signals.
    """
    heading = chunk.get("heading", "")
    text = chunk.get("chunk_text", "")
    combined = f"{heading}\n{text}"

    gui_hits = len(_CHUNK_GUI_RE.findall(combined))
    cli_hits = len(_CHUNK_CLI_RE.findall(combined))

    # Heading is a strong signal — double its weight
    heading_gui = len(_CHUNK_GUI_RE.findall(heading))
    heading_cli = len(_CHUNK_CLI_RE.findall(heading))
    gui_hits += heading_gui
    cli_hits += heading_cli

    if cli_hits > gui_hits:
        return "cli"
    if gui_hits > cli_hits:
        return "gui"
    return "neutral"


# ── Interface-aware score adjustment ─────────────────────────────────────────

# Score bonus/penalty applied to _rerank_score after the cross-encoder.
# These are additive adjustments to cross-encoder scores (range ~[-10, 12]).
_INTERFACE_MATCH_BONUS = 1.5     # bonus when chunk matches query intent
_INTERFACE_MISMATCH_PENALTY = 2.5  # penalty when chunk contradicts query intent


def _apply_interface_bias(
    results: list[dict[str, Any]],
    intent: str,
) -> list[dict[str, Any]]:
    """
    Adjust rerank scores based on CLI/GUI intent match.

    For each chunk:
      - Label it as cli/gui/neutral
      - If intent is "cli" and chunk is "cli" → bonus
      - If intent is "cli" and chunk is "gui" → penalty
      - Vice versa for "gui" intent
      - "neutral" intent → no adjustment

    Results are re-sorted by adjusted score after application.
    Attaches '_interface' (chunk label) and '_interface_adj' (adjustment)
    to each result dict.
    """
    if intent == "neutral":
        for r in results:
            r["_interface"] = _classify_chunk_interface(r)
            r["_interface_adj"] = 0.0
        return results

    preferred = intent  # "cli" or "gui"
    opposite = "gui" if intent == "cli" else "cli"

    for r in results:
        chunk_if = _classify_chunk_interface(r)
        r["_interface"] = chunk_if

        adj = 0.0
        if chunk_if == preferred:
            adj = _INTERFACE_MATCH_BONUS
        elif chunk_if == opposite:
            adj = -_INTERFACE_MISMATCH_PENALTY

        r["_interface_adj"] = adj
        r["_rerank_score"] = r.get("_rerank_score", 0.0) + adj

    # Re-sort by adjusted score
    results.sort(key=lambda r: r["_rerank_score"], reverse=True)
    return results


# ── Mismatch detection ───────────────────────────────────────────────────────

def _detect_interface_mismatch(
    results: list[dict[str, Any]],
    intent: str,
) -> str | None:
    """
    Check whether the final results mostly don't match the query's interface intent.

    Returns a mismatch note string if ≥ half of top results are the
    opposite interface, or None if no mismatch.
    """
    if intent == "neutral" or not results:
        return None

    opposite = "gui" if intent == "cli" else "cli"
    opposite_label = "GUI/web-console" if intent == "cli" else "CLI/command-line"
    wanted_label = "CLI/command-line" if intent == "cli" else "GUI/web-console"

    opposite_count = sum(1 for r in results if r.get("_interface") == opposite)
    neutral_count = sum(1 for r in results if r.get("_interface") == "neutral")
    matched_count = len(results) - opposite_count - neutral_count

    if opposite_count > matched_count and opposite_count >= len(results) // 2:
        return (
            f"The query asks for {wanted_label} instructions, but "
            f"{opposite_count} of {len(results)} retrieved chunks are "
            f"{opposite_label}-oriented. The retrieved evidence may not "
            f"contain the specific {wanted_label} procedure you need."
        )
    return None


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
    Full retrieval pipeline: classify → hybrid search → rerank → threshold
    → interface bias → dedup → mismatch check.

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
            Each result has '_score', '_rerank_score', '_query_type',
            '_interface_intent', '_interface', '_interface_adj', and
            optionally '_interface_mismatch' (str note or None).
        """
        k = top_k or settings.search_top_k
        n = top_n or settings.search_rerank_top_n

        # ── Step 0: Classify query ──────────────────────────────────────
        query_type = classify_query(query)
        interface_intent = detect_interface_intent(query)
        logger.debug(
            "Query classified as '%s', interface='%s': %s",
            query_type, interface_intent, query[:60],
        )

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
        # Ask reranker for more than final n so dedup + interface bias
        # have room to promote matching chunks.
        # When the user has a clear CLI/GUI intent we widen the window
        # substantially, because the cross-encoder often ranks the
        # "wrong" interface highly (same topic, different method).
        if interface_intent != "neutral":
            # Rerank ALL candidates so interface bias can promote
            # CLI/GUI chunks that the cross-encoder under-ranked.
            rerank_n = len(candidates)
        else:
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

        # ── Step 4: Interface bias (after rerank, before dedup) ─────────
        results = _apply_interface_bias(results, interface_intent)
        if interface_intent != "neutral":
            logger.debug(
                "Interface bias '%s': %s",
                interface_intent,
                [(r.get("heading", "")[:40], r["_interface"], r["_interface_adj"])
                 for r in results[:6]],
            )

        # ── Step 5: Deduplicate ─────────────────────────────────────────
        results = _deduplicate(results, settings.dedup_similarity_threshold)

        # ── Step 6: Trim to final n ─────────────────────────────────────
        # When the user has a clear interface preference, return a few
        # extra chunks to improve topic coverage (e.g. both "create" and
        # "extend" when the query asks for both).
        final_n = n + 2 if interface_intent != "neutral" else n
        results = results[:final_n]

        # ── Step 7: Mismatch detection ──────────────────────────────────
        mismatch_note = _detect_interface_mismatch(results, interface_intent)
        if mismatch_note:
            logger.info("Interface mismatch: %s", mismatch_note)

        # ── Step 8: Attach metadata ─────────────────────────────────────
        for r in results:
            r["_query_type"] = query_type
            r["_interface_intent"] = interface_intent
            r["_interface_mismatch"] = mismatch_note

        logger.debug(
            "Retriever '%s': %d candidates → %d final "
            "(reranker=%s, type=%s, interface=%s)",
            query[:40], len(candidates), len(results),
            "on" if self._use_reranker else "off", query_type, interface_intent,
        )

        return results
