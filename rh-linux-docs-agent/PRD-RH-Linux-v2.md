# PRD v2: RAG System for Red Hat Enterprise Linux Documentation

## 1. Overview

A Retrieval-Augmented Generation (RAG) system for Red Hat Enterprise Linux (RHEL) documentation, designed to power a grounded Q&A agent for troubleshooting RHEL systems. The agent retrieves relevant documentation chunks, assembles them into context, and generates answers strictly from evidence — never from its own training data.

**Current scope:** RHEL 9 only. RHEL 8 and 10 support deferred to a future release.

### How it works

1. Documentation HTML pages are scraped from `docs.redhat.com` and cached locally
2. A parser extracts structured sections (headings, code blocks, tables, admonitions) from the HTML
3. A chunker splits sections into 100–800 token pieces, preserving code blocks and section hierarchy
4. An embedder converts each chunk into a 384-dimensional vector using BGE-small-en-v1.5
5. Chunks + vectors are stored in LanceDB (embedded, serverless, file-based)
6. At query time: hybrid retrieval (vector + BM25 + RRF) → cross-encoder reranking → deduplication → score threshold → context assembly → LLM answer generation with mandatory citations

---

## 2. Problem Statement

RHEL administrators and SREs need fast, accurate answers from RHEL documentation when troubleshooting systems. The documentation spans thousands of pages across multiple versions. Manual searching is slow and error-prone.

### Source constraint

RHEL documentation source (AsciiDoc) is not publicly available. The docs are authored internally by Red Hat and published to `docs.redhat.com`. The only practical collection method is HTML scraping of the single-page HTML renderings.

---

## 3. Goals

- Query RHEL 9 documentation with high accuracy
- Ground every answer in retrieved evidence — no hallucination
- Cite every claim with guide title, section heading, and full URL
- Explicitly state uncertainty when evidence is insufficient
- Support air-gapped deployment (all models run locally on CPU)
- Modular pipeline: each stage is independently testable and replaceable

### Non-goals (deferred)

- Multi-version support (RHEL 8, 10) — architecture supports it, not yet populated
- Cross-version comparison tool (`docs_compare`)
- Web UI / Gradio chat interface
- REST API for external tool integration
- pydantic-ai agent framework integration

---

## 4. Implemented Architecture

### 4.1 Data flow

```
docs.redhat.com (HTML-single)
        ↓
    Scraper (httpx + BeautifulSoup)
        ↓
    Parser (recursive section extraction)
        ↓
    Chunker (section-aware, 100–800 tokens)
        ↓
    Embedder (BAAI/bge-small-en-v1.5, 384-dim)
        ↓
    LanceDB (vectors + 22 metadata columns)
        ↓
    HybridSearch (vector + BM25 + RRF fusion)
        ↓
    Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
        ↓
    Score Threshold + Deduplication
        ↓
    Query Classifier (procedure / concept / troubleshooting / reference)
        ↓
    Context Assembly (numbered sources with full citation metadata)
        ↓
    LLM Generation (strict grounding policy)
        ↓
    Answer (text + sources + confidence + query type)
```

### 4.2 Technology stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Scraper | `httpx` + `beautifulsoup4` | Rate-limited (1.5s delay), cached to disk |
| Parser | Custom `parser/html_parser.py` | Recursive `<section>` walking, RHEL 9 HTML structure |
| Chunker | Custom `chunker/splitter.py` | Section-aware, code blocks atomic, tiktoken counting |
| Embedder | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | 384-dim, ~130MB, CPU, ~3k chunks/min |
| Vector DB | `lancedb` | Embedded (no server), IVF-PQ index + Tantivy BM25 FTS index |
| Hybrid search | Custom `search/hybrid.py` | Vector + BM25 + RRF fusion (70/30 default) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB, CPU, lazy-loaded |
| QA engine | Custom `agent/qa.py` | Context assembly + LLM call + offline fallback |
| LLM | `deepseek/deepseek-chat-v3-0324` via OpenRouter | Temperature 0.1, 2000 max tokens |
| CLI | `typer` + `rich` | Demo scripts, evaluation harness |
| Config | `pydantic-settings` | `.env` + environment variables, typed defaults |
| Token counting | `tiktoken` (`cl100k_base`) | Used by chunker for size enforcement |

---

## 5. Pipeline Stages — Implemented Detail

### 5.1 HTML Parser (`parser/html_parser.py`)

**Input:** Cached single-page HTML files from `data/html_cache/rhel9/`

**Process:**
- Strips noise: scripts, styles, nav, boilerplate CSS classes
- Recursively walks `<section>` elements to build a heading hierarchy
- Extracts heading text and level, cleaning RHEL 9–specific "Copy link" spans
- Processes content elements: paragraphs, lists (ordered/unordered with nesting), code blocks (`<pre>`), admonitions (NOTE/WARNING/IMPORTANT/TIP/CAUTION), tables (converted to markdown pipe format)
- Classifies each section as `procedure`, `concept`, or `reference` based on content signals (ordered lists, step patterns, definition lists, table density)

**Output:** `ParsedGuide` object with flat list of `Section` records, each carrying:
- `record_id`, `guide_slug`, `guide_title`, `guide_url`, `section_url`
- `heading`, `heading_level`, `heading_path` (full hierarchy)
- `body_text`, `content_type`
- `product`, `major_version`, `minor_version`, `doc_type`

### 5.2 Chunker (`chunker/splitter.py`)

**Input:** Parsed section records

**Chunking rules:**

| Rule | Value | Rationale |
|------|-------|-----------|
| Minimum size | 100 tokens | Below this, chunks lack context — merge with adjacent |
| Target size | 500 tokens | Large enough for meaningful retrieval, small enough for precision |
| Maximum size | 800 tokens | Hard ceiling — split at paragraph boundaries |
| Code blocks | Never split | Partial code is useless; kept atomic even if >800 tokens |
| Tables | Split by row groups | Header row preserved in each sub-chunk |
| Hierarchy | Carried on every chunk | `heading_path_text` provides section context |

**Segmentation strategy:**
1. `_segment_body()` splits body text into semantic segments: code blocks → tables → paragraph groups
2. `_pack_segments()` greedily packs segments into chunks within token limits
3. Oversized prose is split at paragraph boundaries → list item boundaries → line boundaries
4. Oversized tables are split into row groups, each carrying the header row

**Output:** List of chunk dicts, each with deterministic `chunk_id = "{parent_record_id}/c{chunk_index}"` and 22 metadata fields.

### 5.3 Embedding + Indexing

**Embedder (`indexer/embedder.py`):**
- Model: `BAAI/bge-small-en-v1.5` — 384-dimensional, normalized cosine vectors
- BGE query prefix: `"Represent this sentence for searching relevant passages: "`
- Batch embedding with rich progress bar
- CPU-only, ~130MB model download

**Schema (`indexer/schema.py`):**

22 metadata columns + 1 vector column in PyArrow schema:

| Column | Type | Purpose |
|--------|------|---------|
| `chunk_id` | string | Primary key (deterministic) |
| `parent_record_id` | string | Links to parser section |
| `chunk_index` | int32 | Position within section |
| `product` | string | Always "Red Hat Enterprise Linux" |
| `major_version` | string | "9" |
| `minor_version` | string | e.g., "9.4" |
| `doc_type` | string | e.g., "guide", "release_notes" |
| `guide_slug` | string | URL slug for filtering |
| `guide_title` | string | Human-readable guide name |
| `guide_url` | string | Full guide URL |
| `section_url` | string | Deep-link to specific section |
| `heading` | string | Section heading text |
| `hierarchy` | string | JSON-encoded heading path |
| `heading_path_text` | string | " > "-joined heading path |
| `section_id` | string | HTML section anchor |
| `content_type` | string | procedure / concept / reference |
| `chunk_text` | string | The actual text content |
| `char_count` | int32 | Character count |
| `word_count` | int32 | Word count |
| `has_code_blocks` | bool | Contains code fences |
| `has_tables` | bool | Contains table markup |
| `vector` | fixed_size_list(float32, 384) | Embedding vector |

**Store (`indexer/store.py`):**
- LanceDB embedded database at `data/lancedb/`, table `rhel_docs`
- Upsert pattern: delete by `chunk_id` then insert (idempotent)
- IVF-PQ vector index: `num_partitions = min(256, max(4, sqrt(row_count)))`
- BM25/FTS index on `chunk_text` via Tantivy

**Current corpus:** ~140 RHEL 9 guides → ~28,000 chunks indexed.

### 5.4 Hybrid Retrieval (`search/hybrid.py`)

Three search modes, selectable at query time:

**Vector search:**
- Cosine similarity on BGE-small embeddings
- Query embeddings use BGE instruction prefix
- Prefilter metadata with SQL WHERE clause

**BM25 full-text search:**
- Tantivy-based FTS index on `chunk_text`
- Same metadata prefilter via WHERE clause
- Fallback: if FTS index missing, hybrid mode degrades to vector-only

**Reciprocal Rank Fusion (RRF):**
```
score(doc) = vector_weight / (k + rank_vector) + bm25_weight / (k + rank_bm25)
```
- `k = 60` (smoothing constant, per RRF paper)
- `vector_weight = 0.7`, `bm25_weight = 0.3`
- Merges by `chunk_id` — documents appearing in both lists get both score components

**Metadata filters** (all optional, applied as SQL WHERE prefilter):
- `product`, `major_version`, `minor_version`, `doc_type`, `guide_slug`, `content_type`

### 5.5 Cross-Encoder Reranker (`search/reranker.py`)

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- ~80MB, CPU, lazy-loaded on first call
- Trained on MS MARCO passage ranking
- Scores (query, passage) pairs directly — much more accurate than bi-encoder similarity
- Score range: roughly [-10, +12]; >7 = strong match, 4–7 = moderate, <4 = weak

**Pipeline position:** After hybrid retrieval returns `top_k=20` candidates, the reranker scores all 20 and returns the top results sorted by `_rerank_score`.

**Text truncation:** Chunk text truncated to 2000 chars before cross-encoder (model max ~512 tokens).

**NoOpReranker:** Pass-through class that copies `_score` → `_rerank_score` for no-reranker mode.

### 5.6 Post-Retrieval Filtering (`search/retriever.py`)

The `Retriever` class orchestrates the full pipeline with three post-rerank stages:

**1. Score threshold:**
- Minimum rerank score: `3.0` (configurable via `rerank_score_threshold`)
- Only active when cross-encoder reranker is in use
- Drops weak matches even if `top_n` is not reached
- Example: Oracle RAC query — 7 reranked candidates → 4 dropped below 3.0 → only 3 returned

**2. Deduplication:**
- A chunk is a duplicate if:
  - (a) Same `heading` + same `guide_slug` as a higher-scored chunk, OR
  - (b) Jaccard word-overlap ≥ 0.70 with a higher-scored chunk's text
- Higher-scored chunk always survives
- Example: Firewalld DMZ zone procedure appears in both `configuring_firewalls_and_packet_filters` and `automating_system_administration_by_using_rhel_system_roles` with near-identical text — the duplicate is removed

**3. Final trim:** Results capped at `top_n` (default 4).

**Over-fetch:** Reranker is asked for `top_n + 3` candidates to leave room after dedup/threshold filtering.

### 5.7 Query Classification (`search/retriever.py`)

Simple keyword-regex classifier. Returns one of four types:

| Type | Triggers | Example |
|------|----------|---------|
| `procedure` | how to, configure, set up, install, create, enable, join, mount | "How to configure a static IP?" |
| `troubleshooting` | troubleshoot, debug, fix, error, denied, not working | "Troubleshoot SELinux denials" |
| `concept` | what is, explain, overview, difference between, how does | "What is the default crypto policy?" |
| `reference` | list, show, default values, release notes, changes in | "List RHEL 9.4 release note changes" |

Default when no pattern matches: `procedure` (most RHEL questions are how-to).

The classification is attached to every result as `_query_type` and propagated to the `Answer` object for downstream use (e.g., UI display, analytics, future content_type biasing).

### 5.8 QA Answer Layer (`agent/qa.py`)

**Context assembly (`assemble_context`):**
- Numbers chunks as `[Source 1]`, `[Source 2]`, etc.
- Each chunk header includes: Guide name, Section heading, Content type, Full URL
- Individual chunks truncated at 3000 chars
- Total context capped at 12000 chars
- Returns `(context_string, list[Source])`

**Confidence assessment (`_assess_confidence`):**

Pre-LLM scoring based on retrieval quality:

| Score type | High | Medium | Low | Insufficient |
|------------|------|--------|-----|--------------|
| Cross-encoder | >7 (or >4 with avg_top3 >5) | >4 | >2 | ≤2 |
| RRF/vector (0–1) | >0.5 | >0.3 | >0.005 | ≤0.005 |

The system auto-detects score type by checking `abs(top_score) > 1.5`.

**LLM generation:**
- OpenRouter API (OpenAI-compatible), model: `deepseek/deepseek-chat-v3-0324`
- Temperature: 0.1 (near-deterministic)
- Max tokens: 2000
- System prompt: 9 strict grounding rules (see §6)
- User message: question + retrieved context + explicit no-hallucination instructions

**Offline fallback:** When no API key is set, `_build_offline_answer()` produces a structured answer showing verbatim retrieved context with consistent citations. No commands or procedures are synthesized.

**Answer object:**

```python
@dataclass
class Answer:
    query: str                # Original question
    text: str                 # Generated answer (or offline fallback)
    sources: list[Source]     # Cited sources with guide_title, heading, URL
    confidence: str           # "high" | "medium" | "low" | "insufficient"
    query_type: str           # "procedure" | "concept" | "troubleshooting" | "reference"
    retrieval_time_s: float   # Retrieval + rerank + dedup time
    generation_time_s: float  # LLM call time
    model: str                # Model used (or "offline")
    context_chunks: int       # Number of chunks in context
    raw_context: str          # Full context string (for debugging)
```

---

## 6. Grounding Policy

The system enforces strict evidence-only answers. This is the core quality contract.

### 6.1 System prompt rules

1. **Evidence-only answers.** Every statement must come from retrieved chunks. No commands, paths, CLI options, config directives, or package names may be introduced from the model's training data.

2. **No gap-filling.** If retrieved context discusses a topic but lacks the specific command or step, the model must say so explicitly:
   - *"The retrieved documentation covers [X], but the specific command/procedure for [Y] is not shown in the retrieved evidence."*
   - *"For the exact CLI syntax, consult the full guide linked below."*

3. **Mandatory citations.** Every factual claim must be followed by `[Source N]`. A `**Sources:**` list at the end of every answer includes guide title, section heading, and full URL.

4. **Confidence header.** Every answer begins with one of: `**Confidence: High**`, `**Confidence: Medium**`, `**Confidence: Low**`, `**Confidence: Insufficient**`.

5. **Uncertainty escalation.** When confidence is Medium, Low, or Insufficient, the answer includes a link to the full section URL and to `https://docs.redhat.com`.

6. **Verbatim reproduction.** Code blocks and config snippets from the context are reproduced exactly. No modifications, extensions, or added flags.

7. **Complementary & conflicting sources.** When multiple chunks from different guides address the same question, the model notes this. Conflicting information is flagged with both citations.

8. **Version awareness.** Context is RHEL 9. If the user asks about RHEL 8 or 10, the model states that evidence is RHEL 9–specific.

### 6.2 User message reinforcement

The user message sent to the LLM repeats the core constraints:
- Cite sources as `[Source N]`
- Do NOT introduce any commands, file paths, CLI flags, config directives, or package names not verbatim in the chunks
- If chunks discuss the topic but lack the specific command, say what IS covered and what is NOT
- End with a `**Sources:**` list in consistent format

### 6.3 Citation format

All citations follow this format (in both LLM and offline answers):

```
**Sources:**
1. *Guide Title — Section Heading*
   https://docs.redhat.com/en/documentation/...full URL...
```

The `Source` dataclass provides a `citation_label` property (`"Guide Title — Heading"`) used everywhere for consistency.

---

## 7. Configuration

All settings are centralized in `config.py` using `pydantic-settings`. Loaded from `.env` file or environment variables.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `db_path` | Path | `data/lancedb` | LanceDB storage directory |
| `table_name` | str | `rhel_docs` | LanceDB table name |
| `cache_dir` | Path | `data/html_cache` | Downloaded HTML cache |
| `embedding_model` | str | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `embedding_dim` | int | `384` | Vector dimensions (must match model) |
| `embedding_batch_size` | int | `32` | Chunks per embedding batch |
| `chunk_min_tokens` | int | `150` | Minimum chunk size |
| `chunk_max_tokens` | int | `800` | Maximum chunk size |
| `chunk_target_tokens` | int | `500` | Target chunk size |
| `openrouter_api_key` | str | `""` | OpenRouter API key (from `.env`) |
| `llm_model` | str | `deepseek/deepseek-chat-v3-0324` | LLM for answer generation |
| `search_top_k` | int | `20` | Retrieval candidates per method |
| `search_rerank_top_n` | int | `4` | Final results after reranking |
| `rerank_score_threshold` | float | `3.0` | Minimum cross-encoder score to keep |
| `dedup_similarity_threshold` | float | `0.70` | Jaccard overlap to flag duplicate |
| `rrf_k` | int | `60` | RRF smoothing constant |
| `vector_weight` | float | `0.7` | Vector weight in RRF fusion |
| `scrape_delay` | float | `1.5` | Seconds between HTTP requests |
| `scrape_retries` | int | `3` | Retry count for failed downloads |
| `user_agent` | str | `rh-linux-docs-agent/1.0 (...)` | HTTP User-Agent |
| `web_port` | int | `7933` | Gradio web UI port (future) |

---

## 8. Retrieval Evaluation

### 8.1 Evaluation set

20 test queries across 7 categories, each with expected guide slugs:

| Category | Queries | Example |
|----------|---------|---------|
| Networking | 4 | Static IP with nmcli, network bridge for KVM, firewalld SSH+HTTPS, DNS client |
| Security | 4 | SELinux enforcing, SELinux booleans for httpd, FIPS crypto policies, SSH key auth |
| Storage | 3 | LVM logical volumes, Stratis pools, NFS mount via fstab |
| Containers | 2 | Podman build/run, systemd service for Podman |
| System management | 3 | dnf-automatic, systemd journal, sysctl at boot |
| Identity | 2 | IdM/FreeIPA server setup, Active Directory join |
| Virtualization + misc | 2 | KVM virt-install, RHEL 9.4 release notes |

### 8.2 Metrics

| Metric | Description |
|--------|-------------|
| Hit@1 | Relevant result at rank 1 |
| Hit@3 | Relevant result in top 3 |
| Hit@5 | Relevant result in top 5 |
| MRR | Mean Reciprocal Rank (1/rank of first relevant result, averaged) |

### 8.3 Results (hybrid + reranker)

Measured with `scripts/eval_retrieval.py` on the 20-query set:

| Metric | hybrid + reranker |
|--------|-------------------|
| Hit@1 | 17/20 (85%) |
| Hit@3 | 19/20 (95%) |
| Hit@5 | 20/20 (100%) |
| MRR | 0.917 |
| Avg query time | ~1.5s |

The 3 queries not at rank 1 still found relevant content in top 3, from different (but valid) guides than the expected slugs. The evaluation set was broadened to account for correct content appearing in multiple guides.

---

## 9. Project Structure

```
rh-linux-docs-agent/
├── src/rh_linux_docs_agent/
│   ├── config.py                  # Centralized pydantic-settings configuration
│   ├── scraper/
│   │   ├── discovery.py           # Guide URL discovery from landing pages
│   │   ├── fetcher.py             # HTML download + disk caching
│   │   └── parser.py              # Recursive section extraction from HTML
│   ├── chunker/
│   │   └── splitter.py            # Section-aware token-bounded chunking
│   ├── indexer/
│   │   ├── embedder.py            # BGE-small embedding wrapper
│   │   ├── schema.py              # PyArrow schema + record converters
│   │   └── store.py               # LanceDB table operations + index building
│   ├── search/
│   │   ├── hybrid.py              # Vector + BM25 + RRF fusion search
│   │   ├── reranker.py            # Cross-encoder reranker + NoOpReranker
│   │   └── retriever.py           # Full pipeline: classify → search → rerank → dedup
│   └── agent/
│       └── qa.py                  # QA engine: context assembly + LLM + citations
├── scripts/
│   ├── scrape_guides.py           # CLI: scrape HTML from docs.redhat.com
│   ├── parse_guides.py            # CLI: parse cached HTML into section records
│   ├── chunk_sections.py          # CLI: chunk section records
│   ├── index_docs.py              # CLI: embed + index chunks into LanceDB
│   ├── search_demo.py             # CLI: 5 example search queries
│   ├── eval_retrieval.py          # CLI: 20-query retrieval evaluation
│   ├── qa_demo.py                 # CLI: QA demo with rich output
│   └── qa_showcase.py             # CLI: QA showcase (offline mode)
├── data/
│   ├── html_cache/rhel9/          # Cached HTML pages
│   ├── parsed/rhel9/              # Parsed section JSON files
│   ├── chunked/rhel9/             # Chunked JSON files
│   └── lancedb/                   # LanceDB database files
├── pyproject.toml                 # Dependencies + project metadata
├── .env                           # API keys (not committed)
└── .gitignore
```

---

## 10. Current Scope & Limitations

### 10.1 Current scope

- **RHEL 9 only.** The schema and filters support multi-version (`major_version`, `minor_version`), but only RHEL 9 content is ingested and tested.
- **~140 guides, ~28,000 chunks** indexed.
- **Offline-capable retrieval.** Embedding model and reranker run on CPU with no API calls.
- **LLM requires API key.** Answer generation uses OpenRouter. Without a key, the system falls back to showing retrieved context directly with no synthesized text.

### 10.2 Known limitations

1. **No web UI.** The Gradio chat interface from the original PRD is not yet built. Interaction is via CLI scripts and Python API.

2. **No pydantic-ai agent.** The original PRD specified a pydantic-ai agent with `docs_search` and `docs_compare` tools. The current QA engine is a direct pipeline (not an agentic loop). This is simpler and sufficient for the current scope.

3. **No `docs_compare` tool.** Cross-version comparison requires RHEL 8/10 content. Deferred.

4. **No REST API.** The search functionality is not yet exposed as an HTTP API.

5. **Query classification is keyword-based.** The classifier uses regex patterns, not a trained model. It covers common patterns well but may misclassify ambiguous queries. The classification is currently informational (attached to results) — it does not yet hard-filter by `content_type`.

6. **Deduplication is text-level only.** Jaccard word-overlap catches near-identical chunks but not semantic duplicates where the same concept is expressed in different words.

7. **Score threshold is static.** The `rerank_score_threshold = 3.0` works well for the current corpus and cross-encoder model but would need recalibration if the reranker model changes.

8. **Confidence assessment is heuristic.** The pre-LLM confidence scoring uses score thresholds, not a trained relevance classifier. It can be fooled by high-scoring but off-topic chunks.

9. **No incremental re-indexing.** Adding new guides requires re-running the full indexing pipeline for affected guides. There is no change-detection or delta-update mechanism.

10. **Single LLM provider.** Currently hardcoded to OpenRouter. No local LLM fallback (Ollama, vLLM) is configured, though the offline answer builder provides a no-LLM alternative.

11. **No unit test suite.** The system is validated through the 20-query evaluation harness and manual demo scripts, but there are no pytest unit tests for individual modules.

---

## 11. Changes vs v1

| Aspect | PRD v1 (planned) | PRD v2 (implemented) |
|--------|-------------------|----------------------|
| **Scope** | RHEL 8, 9, 10 simultaneously | RHEL 9 only (multi-version deferred) |
| **Agent framework** | pydantic-ai with tool registration | Direct QA pipeline (no agentic loop) |
| **Agent tools** | `docs_search` + `docs_compare` | `QAEngine.ask()` + `QAEngine.retrieve_only()` |
| **Web UI** | Gradio via pydantic-ai[web] | Not yet implemented |
| **LLM model** | `deepseek-chat-v3.1` | `deepseek-chat-v3-0324` (via OpenRouter) |
| **Reranker** | Not in v1 PRD | Cross-encoder `ms-marco-MiniLM-L-6-v2` added |
| **Score threshold** | Not in v1 PRD | Minimum rerank score 3.0 to drop weak matches |
| **Deduplication** | Not in v1 PRD | Jaccard overlap + heading match dedup |
| **Query classification** | Not in v1 PRD | Keyword-regex classifier (4 types) |
| **Context window** | `search_rerank_top_n = 5` | `search_rerank_top_n = 4` (tighter focus) |
| **Grounding policy** | "Always cite guide title and URL" | 9-rule strict grounding: no gap-filling, verbatim reproduction, confidence header, mandatory citations with consistent format |
| **Citation format** | URL only | `*Guide Title — Section Heading*` + full URL |
| **Confidence scoring** | Not in v1 PRD | Pre-LLM heuristic: high/medium/low/insufficient |
| **Offline fallback** | Not in v1 PRD | Structured answer from raw context when no API key |
| **Retrieval evaluation** | "20–30 test queries" (planned) | 20 queries, 7 categories, measured Hit@1/3/5 + MRR |
| **LanceDB schema** | 10 fields | 22 metadata fields + vector |
| **Chunk ID** | `"{version}::{guide}::{section_path}::{chunk_idx}"` | `"{parent_record_id}/c{chunk_index}"` (deterministic) |
| **REST API** | Planned (Milestone 9) | Deferred |
| **Air-gapped LLM** | "locally-hosted model (vLLM, Ollama)" | Offline fallback shows raw context; local LLM integration deferred |
| **Project timeline** | 12.5 days across 11 milestones | Milestones 1–6 + 11 completed; 7–10 deferred |

### What was preserved from v1

- HTML-single scraping strategy (not PDF)
- `BAAI/bge-small-en-v1.5` embedding model (384-dim, CPU)
- LanceDB as the vector database
- Hybrid vector + BM25 + RRF search with the same default parameters (`rrf_k=60`, `vector_weight=0.7`, `search_top_k=20`)
- Section-aware chunking with atomic code blocks (100–800 token range)
- `pydantic-settings` for configuration
- `typer` + `rich` for CLI
- Project directory structure (src layout, scripts/, data/)
- Cache-first scraping with rate limiting
- All core configuration defaults (db_path, table_name, cache_dir, scrape_delay, embedding_batch_size)

---

## 12. Air-Gapped Deployment

Architecture supports air-gapped use. All models run on CPU with no API calls except LLM generation.

**Pre-download on connected machine:**
1. HTML cache: `data/html_cache/rhel9/` (~200+ HTML files)
2. Embedding model: `BAAI/bge-small-en-v1.5` (~130MB from HuggingFace)
3. Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB)
4. Python packages: `pip download -d ./wheels -e .`

**Transfer + run offline:**
1. Extract HTML cache, models, wheels
2. `pip install --no-index --find-links=./wheels -e .`
3. Run ingestion pipeline (no network needed)
4. For LLM generation: use local model (Ollama/vLLM) or accept offline fallback

The offline fallback (`_build_offline_answer`) produces a fully usable answer from retrieved context alone — no LLM required.

---

## 13. Future Work

Ordered by priority:

1. **RHEL 8 + 10 ingestion** — Populate additional versions, enable `docs_compare` cross-version tool
2. **Web UI** — Gradio chat interface for interactive Q&A
3. **Local LLM support** — Ollama or vLLM backend for fully air-gapped answer generation
4. **Unit test suite** — pytest tests for parser, chunker, embedder, search, QA
5. **REST API** — Expose search and QA as HTTP endpoints
6. **Query classification → retrieval bias** — Use classified query type to boost matching `content_type` in retrieval
7. **Incremental re-indexing** — Detect changed guides and re-ingest only deltas
8. **Semantic deduplication** — Embedding-based duplicate detection (not just word overlap)
9. **pydantic-ai agent** — Agentic loop with tool calling for multi-step queries
10. **Search analytics** — Log queries and results to identify documentation coverage gaps
11. **Fedora / CentOS Stream docs** — Expand beyond RHEL
12. **Knowledge articles** — Integrate Red Hat Knowledgebase (requires subscription API)
