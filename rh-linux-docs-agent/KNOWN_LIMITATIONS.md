# Known Limitations

Current limitations of the RHEL 9 Documentation Agent v1 release.

---

## Scope

**1. RHEL 9 only.**
The schema and metadata filters support multi-version queries (`major_version`, `minor_version`), but only RHEL 9 content is ingested, indexed, and tested. RHEL 8 and 10 are deferred.

**2. No cross-version comparison.**
The `docs_compare` tool from the original PRD is not implemented. It requires content from multiple RHEL versions.

---

## Retrieval

**3. Query classification is keyword-based.**
The classifier uses regex patterns to detect procedure/troubleshooting/concept/reference queries. It covers common patterns well but can misclassify ambiguous queries. Currently informational only — it does not hard-filter retrieval results by `content_type`.

**4. Deduplication is text-level only.**
Near-duplicate detection uses Jaccard word-overlap (threshold 0.70). It catches identical or near-identical chunks across guides but misses semantic duplicates where the same concept is stated in different words.

**5. Score threshold is static.**
The `rerank_score_threshold = 3.0` was tuned for the current corpus and cross-encoder model (`ms-marco-MiniLM-L-6-v2`). Changing the reranker model or significantly expanding the corpus would require recalibration.

**6. Confidence assessment is heuristic.**
Pre-LLM confidence scoring uses rerank score thresholds, not a trained relevance classifier. A high-scoring but off-topic chunk can inflate confidence.

---

## Answer generation

**7. LLM requires external API.**
Answer generation depends on OpenRouter (deepseek-chat-v3-0324). No local LLM backend (Ollama, vLLM) is configured. The offline fallback shows structured retrieved context but does not synthesize a natural-language answer.

**8. Single LLM provider.**
The `_call_llm` function is hardcoded to OpenRouter's API endpoint. Switching to a different provider requires code changes, not just configuration.

**9. No streaming.**
LLM responses are returned as a single block after generation completes. There is no streaming/incremental display.

---

## Infrastructure

**10. No web UI.**
Interaction is via CLI scripts and Python API only. The Gradio chat interface from the original PRD is not implemented.

**11. No REST API.**
Search and QA are not exposed as HTTP endpoints. External tools cannot call the system over the network.

**12. No incremental re-indexing.**
Adding or updating guides requires re-running the indexing pipeline for those guides. There is no change-detection or delta-update mechanism.

**13. No unit test suite.**
The system is validated through the 20-query evaluation harness and manual demo scripts. There are no pytest unit tests for individual modules.

---

## Parser

**14. Tied to RHEL 9 HTML structure.**
The parser was built for the specific HTML structure of `docs.redhat.com` as of 2025-2026. Changes to their HTML templates (CSS classes, section nesting, admonition markup) would require parser updates.

**15. Images are skipped.**
The parser drops all `<img>` elements. Diagrams, screenshots, and architecture figures in the documentation are not captured.

**16. Some tables lose fidelity.**
Complex HTML tables (nested tables, multi-row spans, colspans) are converted to a simplified markdown pipe format. Layout-heavy tables may lose structural information.
