# Changelog

## v1.0.0 — 2026-03-28

First production-ready release. RHEL 9 documentation agent with hybrid retrieval, cross-encoder reranking, and strict evidence-only answer generation.

### Delivered

**HTML scraper + parser**
- Guide discovery from `docs.redhat.com` landing pages
- HTML-single page download with disk caching and rate limiting (1.5s delay)
- Recursive `<section>` walking to extract headings, paragraphs, code blocks, tables, admonitions (NOTE/WARNING/IMPORTANT/TIP/CAUTION), ordered and unordered lists
- Content classification into procedure / concept / reference
- ~140 RHEL 9 guides parsed

**Chunker**
- Section-aware splitting with 100-800 token bounds (target 500)
- Code blocks kept atomic (never split across chunks)
- Tables split by row groups with header preservation
- Every chunk carries full heading hierarchy context
- Deterministic chunk IDs: `{parent_record_id}/c{chunk_index}`
- ~28,000 chunks produced from RHEL 9 corpus

**Embedding + indexing**
- BAAI/bge-small-en-v1.5 embedder (384-dim, CPU, ~130MB)
- LanceDB embedded database with 22 metadata columns + vector
- IVF-PQ vector index + Tantivy BM25 full-text search index
- Idempotent upsert (delete by chunk_id, then insert)
- PyArrow schema with typed columns for all metadata fields

**Hybrid retrieval**
- Vector search (cosine similarity) + BM25 keyword search
- Reciprocal Rank Fusion: 70% vector / 30% BM25, k=60
- Metadata prefiltering: product, major_version, minor_version, doc_type, guide_slug, content_type
- Graceful degradation: falls back to vector-only if BM25 index unavailable

**Cross-encoder reranker**
- cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU, lazy-loaded)
- Scores (query, passage) pairs for accurate relevance ranking
- NoOpReranker pass-through for benchmarking without reranker

**Post-retrieval filtering**
- Score threshold: minimum rerank score 3.0 (drops weak matches)
- Deduplication: same heading + guide, or Jaccard word-overlap >= 0.70
- Over-fetch (top_n + 3) to leave room after filtering
- Final trim to top 4 chunks

**Query classification**
- Keyword-regex classifier: procedure, troubleshooting, concept, reference
- Attached to every result and answer for downstream use

**QA answer engine**
- Context assembly with numbered [Source N] headers including guide, heading, type, URL
- Pre-LLM confidence assessment (high/medium/low/insufficient) based on rerank scores
- LLM generation via OpenRouter (deepseek-chat-v3-0324, temperature 0.1)
- Offline fallback: structured verbatim context when no API key is set
- Consistent citation format: `*Guide Title -- Section Heading*` + full URL

**Strict grounding policy**
- 9-rule system prompt: evidence-only, no gap-filling, mandatory citations, confidence header, verbatim reproduction, version awareness
- User message reinforcement: explicit no-hallucination instructions
- Verified: firewalld query produces no invented CLI commands

**Retrieval evaluation**
- 20-query test set across 7 categories
- Hit@1: 85%, Hit@3: 95%, Hit@5: 100%, MRR: 0.917
- Evaluation script with per-query relevance tracking and summary metrics

**CLI tools**
- `ingest.py` — full pipeline (scrape + parse + chunk + embed + store)
- `scrape.py` — HTML download only
- `parse_docs.py` — parse cached HTML into section JSON
- `chunk_docs.py` — chunk sections into embedding-ready pieces
- `index_docs.py` — embed + store in LanceDB
- `search_demo.py` — 5 example search queries
- `search_cli.py` — interactive terminal search
- `eval_retrieval.py` — 20-query retrieval evaluation
- `qa_demo.py` — QA demo with rich output
- `qa_showcase.py` — QA showcase (offline mode)

**Configuration**
- Centralized pydantic-settings with .env support
- All search parameters, model paths, thresholds configurable via environment variables

### Not included in v1

- Web UI (Gradio chat interface)
- REST API
- pydantic-ai agent framework
- RHEL 8 / RHEL 10 content
- Cross-version comparison tool
- Unit test suite
- Local LLM integration (Ollama / vLLM)
- Incremental re-indexing
