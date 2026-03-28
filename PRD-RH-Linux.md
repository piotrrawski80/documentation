# PRD: RAG System for Red Hat Enterprise Linux Documentation

## 1. Overview

Build a RAG (Retrieval-Augmented Generation) system for Red Hat Enterprise Linux (RHEL) documentation to power an investigation agent for troubleshooting RHEL systems. The system supports multi-version documentation (RHEL 8, 9, 10), high search accuracy, and air-gapped deployment.

### What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It is a technique where:
1. Documentation is broken into small pieces ("chunks") and stored in a vector database
2. When a user asks a question, the system searches the database for the most relevant chunks
3. Those chunks are passed as context to an LLM (large language model) which generates an accurate answer grounded in the actual documentation

This means the LLM doesn't hallucinate — it answers based on real Red Hat docs.

## 2. Problem Statement

RHEL administrators and SREs need fast, accurate answers from RHEL documentation when troubleshooting systems. The documentation spans thousands of pages across multiple versions with significant differences between major releases (8 vs 9 vs 10). Manual searching is slow and error-prone.

### Documentation Source Constraints

Unlike some Red Hat products that have public git repos with AsciiDoc source, RHEL documentation source is **not publicly available**. The docs are authored internally by Red Hat and published to `docs.redhat.com`. Available collection methods are HTML scraping or PDF download.

## 3. Goals

- Query RHEL documentation across versions 8, 9, and 10
- Compare how procedures, configurations, or features differ between RHEL versions
- Support version-specific and cross-version queries
- High search quality and accuracy
- Run in air-gapped environments
- Expose search functionality via API for external tool integration

## 4. Implementation Notes

### 4.1 Audience

The implementer of this project may have **limited Python and programming experience**. The PRD should be treated as a detailed guide. When building this system with AI coding assistants (Claude Code, Cursor, etc.), the assistant should:

- Explain code decisions with comments — not just what the code does, but **why**
- Prefer simple, readable code over clever abstractions
- Add docstrings to every class and function explaining purpose, parameters, and return values
- Include inline comments for non-obvious logic
- Use type hints on all function signatures
- Provide clear error messages that explain what went wrong and how to fix it
- Log progress at each pipeline stage so the user can see what's happening
- Avoid deeply nested code — prefer early returns and flat structures

### 4.2 Python Virtual Environment (venv) — Required

This project **must** use a Python virtual environment (`venv`). A venv is an isolated Python installation that keeps this project's packages separate from the system Python and other projects. This prevents version conflicts and keeps the system clean.

#### What is a venv?

A virtual environment is like a sandbox for Python packages. Without it, installing packages with `pip` puts them in the global Python installation, which can:
- Break other Python programs on your system
- Cause version conflicts between projects
- Make it impossible to reproduce the exact environment on another machine

#### venv Rules

- The venv **must** be created in the project root directory as `venv/`
- The venv directory **must** be listed in `.gitignore` (never commit it to git)
- All Python commands (running scripts, installing packages) **must** use the venv's Python, not the system Python
- The `pyproject.toml` file defines all dependencies — `pip install -e .` inside the venv installs everything

#### How to Create and Use the venv

```bash
# Step 1: Create the virtual environment (one-time setup)
python3 -m venv venv

# Step 2: Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Step 3: Install the project and all its dependencies
pip install -e .

# Step 4: Verify it works
python -c "import rh_linux_docs_agent; print('OK')"
```

After activation, your terminal prompt will show `(venv)` at the beginning, confirming you're inside the virtual environment. All `python` and `pip` commands will now use the venv's versions.

**If you close your terminal**, you need to re-activate the venv with `source venv/bin/activate` before running any project commands.

#### Using venv Without Activating

You can also run commands directly without activating by using the full path to the venv's Python:

```bash
# Instead of activating + running:
venv/bin/python scripts/ingest.py --version 9
venv/bin/python -m rh_linux_docs_agent.agent.app
```

This is useful for scripts, cron jobs, and IDE configurations where activation isn't practical.

### 4.3 pyproject.toml — The Project Configuration File

The `pyproject.toml` file is the single source of truth for:
- Project name and version
- Python version requirement (>=3.13)
- All package dependencies and their minimum versions
- How the project is installed (`pip install -e .`)

When adding new dependencies, add them to `pyproject.toml` under `[project.dependencies]`, then run `pip install -e .` again inside the venv to install them.

### 4.4 Environment Variables and .env File

Sensitive configuration (API keys, credentials) goes in a `.env` file in the project root. This file **must never be committed to git** — add it to `.gitignore`.

```env
# .env — create this file manually
OPENROUTER_API_KEY=your-api-key-here
```

The project uses `pydantic-settings` to automatically load this file. The `Settings` class in `config.py` reads from `.env` and provides typed access to all configuration.

## 5. Documentation Collection Strategy

### 5.1 Option Analysis

| Option | Source | Pros | Cons |
|--------|--------|------|------|
| **A. HTML-single scraping** | `docs.redhat.com` single-page HTML | Structured headings, code blocks preserved, clean text | Requires scraper, rate limiting, ~200+ guides per version |
| **B. PDF download + extraction** | `docs.redhat.com` PDFs | Bulk download tool exists ([redhat-docs-download](https://github.com/gwojcieszczuk/redhat-docs-download)), offline-friendly | PDF text extraction lossy (tables, code blocks), layout issues |
| **C. Hybrid (HTML primary, PDF fallback)** | Both | Best coverage | More complex pipeline |

### 5.2 Recommendation: Option A — HTML-single scraping

Single-page HTML preserves document structure (headings, code blocks, tables, admonitions) far better than PDF extraction. The URL pattern is predictable:

```
https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/{VERSION}/html-single/{guide_name}/index
```

**Why HTML over PDF?**
- HTML has real headings (`<h1>`, `<h2>`, etc.) that map directly to sections — PDFs lose this structure
- Code blocks in HTML are clearly marked with `<pre>` tags — PDFs just render them as text with a different font
- Tables in HTML have rows and columns — PDF table extraction is notoriously unreliable
- HTML is the canonical format — the PDFs are generated from the same source, so HTML is closer to the original

### 5.3 Collection Pipeline

```
1. Discover guide list       → Scrape product landing page for each version
2. Download HTML-single      → Fetch each guide's single-page HTML
3. Parse & extract sections  → BeautifulSoup / lxml to extract structured content
4. Cache locally             → Store raw HTML + extracted sections on disk
5. Version tag               → Tag all content with RHEL version
```

**Step-by-step explanation:**

1. **Discover** — Visit the RHEL docs landing page (e.g., `docs.redhat.com/en/documentation/red_hat_enterprise_linux/9`) and extract all guide URLs. Each guide covers a topic like "Configuring networking" or "Managing storage devices".

2. **Download** — For each guide, fetch its "single-page" HTML version. This is one long HTML page containing the entire guide (all chapters on one page). This is better for parsing than the multi-page version where each chapter is a separate page.

3. **Parse** — Use BeautifulSoup (a Python HTML parsing library) to walk through the HTML and extract structured content: headings, paragraphs, code blocks, tables, lists. Strip out navigation elements, footers, and other boilerplate.

4. **Cache** — Save the raw HTML files to disk (`data/html_cache/9/configuring_networking.html`). This way you don't have to re-download when re-processing, and it enables offline/air-gapped ingestion.

5. **Version tag** — Tag every extracted piece of content with its RHEL version number so the search system can filter by version later.

## 6. Target Versions

| Version | Status | Priority | Notes |
|---------|--------|----------|-------|
| RHEL 10 | Full Support (GA May 2025) | High | Latest release, most relevant |
| RHEL 9  | Full Support | High | Widely deployed in production |
| RHEL 8  | Maintenance Support (until May 2029) | Medium | Legacy but still common |
| RHEL 7  | Extended Life Phase (maintenance ended June 2024) | Low / Skip | Only if specifically needed |

## 7. Architecture

### 7.1 High-Level Data Flow

```
docs.redhat.com (HTML) → Scraper → Parser → Chunker → Embedder → LanceDB
                                                                      ↓
                                                      Web UI ← Agent (pydantic-ai)
```

**How the pieces fit together:**

1. **Scraper** downloads HTML pages from `docs.redhat.com` and caches them on disk
2. **Parser** reads the cached HTML and extracts structured sections (headings + content)
3. **Chunker** splits large sections into smaller pieces (chunks) of 150–800 tokens each, keeping related content together
4. **Embedder** converts each text chunk into a 384-dimensional vector (a list of 384 numbers that represent the meaning of the text). Similar texts produce similar vectors.
5. **LanceDB** stores all chunks with their vectors in a local database (just files on disk, no server needed)
6. **Agent** receives user questions, searches the database for relevant chunks, and uses an LLM to generate answers based on those chunks
7. **Web UI** provides a chat interface where users type questions and see answers

### 7.2 Technology Stack

| Component | Technology | What it does |
|-----------|-----------|--------------|
| Scraper | `httpx` + `beautifulsoup4` | `httpx` is an HTTP client (downloads web pages). `beautifulsoup4` parses HTML into a tree you can navigate. |
| Parser | Custom HTML parser | Purpose-built code to extract sections, code blocks, and tables from Red Hat's specific HTML structure. |
| Chunker | Section-aware splitter | Splits parsed content into chunks of 150–800 tokens. Keeps sections together, never splits mid-code-block. |
| Embedder | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | A small (130MB) AI model that converts text into 384-dimensional vectors. Runs locally, no API needed. |
| Vector DB | `lancedb` | Embedded vector database. Stores vectors and metadata as files on disk. No server to install or manage. |
| Search | Hybrid: vector + BM25 + RRF | Combines semantic search (meaning-based) with keyword search (exact matches) for best results. |
| Agent | `pydantic-ai` with `deepseek-chat-v3.1` via OpenRouter | The AI agent framework. Takes user questions, searches docs, and generates answers using an LLM. |
| Web UI | `pydantic-ai[web]` (Gradio) | Auto-generated chat web interface. User types a question, agent responds with answers + citations. |
| CLI tools | `typer` + `rich` | `typer` builds command-line interfaces. `rich` makes terminal output pretty (colors, tables, progress bars). |
| Config | `pydantic-settings` | Loads configuration from `.env` files and environment variables with type validation. |
| Token counting | `tiktoken` | Counts tokens in text (used by the chunker to know when a chunk is big enough). |

### 7.3 Key Concepts Explained

**Tokens** — LLMs don't read words, they read "tokens". A token is roughly ¾ of a word. "Configuring NetworkManager" ≈ 3 tokens. The chunker uses token counts to size chunks appropriately.

**Embeddings / Vectors** — An embedding is a list of numbers (e.g., 384 numbers) that represents the meaning of a text. Texts with similar meanings have similar numbers. This is how semantic search works: convert the question to a vector, find stored vectors that are closest to it.

**Cosine similarity** — The math used to measure how similar two vectors are. A score of 1.0 means identical, 0.0 means unrelated.

**BM25** — A traditional keyword search algorithm (like a smart version of Ctrl+F). Good at finding exact matches for commands, file paths, and package names.

**RRF (Reciprocal Rank Fusion)** — A technique to combine results from two different search methods (vector + BM25). Takes the best results from both and merges them into one ranked list.

**LLM (Large Language Model)** — The AI that reads the retrieved chunks and generates a human-readable answer. In this project, we use `deepseek-chat-v3.1` via the OpenRouter API.

### 7.4 Components

Each component is a Python module (a `.py` file) inside the `src/rh_linux_docs_agent/` package:

| Module | File | Purpose |
|--------|------|---------|
| Guide discovery | `scraper/discovery.py` | Visits the RHEL docs landing page and extracts all guide URLs and titles for a given version |
| HTML fetcher | `scraper/fetcher.py` | Downloads HTML-single pages, caches them to disk, respects rate limits |
| HTML parser | `parser/html_parser.py` | Extracts sections, code blocks, tables, and admonitions from cached HTML files |
| Parser models | `parser/models.py` | Data classes: `ParsedGuide` (one guide's content), `Section` (heading + content) |
| Chunk splitter | `chunker/splitter.py` | Splits parsed sections into chunks of 150–800 tokens, preserves code blocks and heading hierarchy |
| Chunk model | `chunker/models.py` | Data class: `Chunk` (text + version + guide + section hierarchy + token count) |
| Embedder | `indexer/embedder.py` | Loads the BGE-small model, converts text chunks into 384-dim vectors in batches |
| DB schema | `indexer/schema.py` | Defines the LanceDB table schema (columns and types) and converts chunks to DB records |
| DB store | `indexer/store.py` | Creates/opens the LanceDB table, inserts records, deletes by version, lists versions |
| Hybrid search | `search/hybrid.py` | Runs vector search + BM25 full-text search, fuses results with RRF, returns ranked results |
| Agent | `agent/agent.py` | Defines the pydantic-ai agent with system prompt and tool registrations |
| Web app | `agent/app.py` | Starts the Gradio web chat UI on a local port |
| Agent tools | `agent/tools.py` | `docs_search()` and `docs_compare()` functions the agent calls to query the database |
| Config | `config.py` | `Settings` class with all configuration (paths, model names, search parameters) |

## 8. Data Model

### 8.1 LanceDB Schema

Every chunk stored in the database has these fields:

```python
{
    "id": str,                 # Unique ID: "{version}::{guide}::{section_path}::{chunk_idx}"
    "version": str,            # RHEL version: "8", "9", or "10"
    "guide": str,              # Guide slug: e.g., "configuring_networking"
    "guide_title": str,        # Human-readable: e.g., "Configuring and managing networking"
    "section_hierarchy": str,  # JSON array of heading path: ["Chapter 5", "5.1 Configuring...", "5.1.2 Adding..."]
    "content_type": str,       # One of: "procedure", "concept", "reference"
    "text": str,               # The actual chunk text content
    "vector": list[float],     # 384-dimensional embedding vector (list of 384 floating-point numbers)
    "token_count": int,        # Number of tokens in the text (used for debugging/tuning)
    "url": str,                # Full URL back to docs.redhat.com for citation
}
```

### 8.2 Field Explanations

- **`id`** — A unique identifier for each chunk. Built from version + guide + section + chunk number. Used to update/delete specific chunks. Example: `"9::configuring_networking::ch5::s1::0"`
- **`version`** — The RHEL major version. Used for filtering queries to a specific version.
- **`guide`** — The URL slug of the guide (the part after `/html-single/` in the URL). Used to group chunks by guide.
- **`guide_title`** — The human-readable title of the guide, shown in search results so users know where the chunk came from.
- **`section_hierarchy`** — A JSON-encoded list of heading titles from the top of the document down to this chunk's section. Provides context about where in the guide this chunk lives.
- **`content_type`** — Whether the chunk describes a procedure (step-by-step), a concept (explanation), or a reference (tables, lists of options). Helps the agent understand what kind of content it's looking at.
- **`text`** — The actual text content of the chunk. This is what gets embedded into a vector and what the LLM reads.
- **`vector`** — The 384-dimensional embedding. This is the numeric representation of the text used for semantic search. Generated by the BGE-small model.
- **`token_count`** — How many tokens the chunk's text contains. Useful for monitoring chunk sizes and tuning the splitter.
- **`url`** — The full URL to the section on `docs.redhat.com`. Included in agent responses so users can verify the source.

## 9. HTML Parsing Strategy

### 9.1 Document Structure

RHEL single-page HTML follows a consistent structure. The parser needs to understand this hierarchy:

```html
<div class="book" or "article">
  <div class="titlepage">...</div>              <!-- Guide title -->
  <div class="chapter">
    <div class="titlepage"><h2>Chapter Title</h2></div>
    <div class="section">
      <div class="titlepage"><h3>Section Title</h3></div>
      <p>Content paragraph...</p>
      <div class="itemizedlist">                <!-- Bulleted list -->
        <ul><li>Item 1</li><li>Item 2</li></ul>
      </div>
      <pre class="programlisting">             <!-- Code block -->
        # dnf install httpd
      </pre>
      <div class="note">                        <!-- Admonition -->
        <p>Important note here...</p>
      </div>
    </div>
  </div>
</div>
```

### 9.2 Extraction Rules

| Element | How to handle | Why |
|---------|--------------|-----|
| **Headings** (`<h1>`–`<h5>`) | Extract text, track nesting level to build section hierarchy | Provides context for each chunk |
| **Code blocks** (`<pre>`, `<code>`) | Preserve content verbatim, never split across chunks | Code must stay intact to be useful |
| **Tables** (`<table>`) | Convert to markdown-style text (pipes and dashes) | Tables contain structured info like config options |
| **Admonitions** (`.note`, `.warning`, `.important`) | Extract with type prefix, e.g., "WARNING: ..." | These contain critical troubleshooting info |
| **Lists** (`<ul>`, `<ol>`) | Flatten to text with bullet/number markers | Lists of steps, options, or requirements |
| **Links** (`<a>`) | Strip the tag but keep the anchor text | The text matters, the URL doesn't |
| **Navigation / boilerplate** | Strip entirely (header, footer, TOC, breadcrumbs, legal notices) | This is noise that would pollute search results |
| **Images** (`<img>`) | Skip (optionally log alt text) | Images can't be embedded as text |

### 9.3 Parsing Edge Cases to Handle

- **Nested sections** — Sections can nest 4–5 levels deep. Track the full heading path.
- **Empty sections** — Some sections only contain sub-sections with no content of their own. Skip these.
- **Inline code** — Short code snippets inside paragraphs (`<code>systemctl restart httpd</code>`) should be kept inline.
- **Long code blocks** — Some code blocks are very long (200+ lines). These should be kept as single chunks even if they exceed the normal chunk size.
- **Procedure steps** — Steps in numbered lists (`<ol>`) are sequential and should stay together when possible.

## 10. Scraping Strategy

### 10.1 Guide Discovery

Each RHEL version's landing page lists all available guides:

```
https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/{VERSION}
```

Parse this page to extract all guide slugs and titles. A "guide" is one documentation book, like:
- "Configuring and managing networking"
- "Managing storage devices"
- "Security hardening"
- "9.5 Release Notes"

### 10.2 Rate Limiting & Caching

- **Respect `robots.txt`** — Check `docs.redhat.com/robots.txt` before scraping and obey its rules
- **Add delays** between requests: 1–2 seconds between page downloads to avoid being blocked
- **Cache all HTML** to `data/html_cache/{version}/{guide_slug}.html` — once downloaded, never re-download unless forced
- **Incremental updates** — Skip guides already in cache. Provide a `--force-refresh` flag to re-download
- **User-Agent header** — Set a descriptive User-Agent: `rh-linux-docs-agent/1.0 (documentation indexer)`
- **Retry with backoff** — If a request fails (HTTP 429, 500, 503), wait and retry up to 3 times with increasing delay
- **Progress logging** — Log each guide being downloaded with a progress counter: `[42/150] Downloading: configuring_networking`

### 10.3 Estimated Scale

| Version | Guides (approx.) | Est. Chunks | Est. Embedding Time |
|---------|-------------------|-------------|---------------------|
| RHEL 10 | ~80 | ~15,000 | ~5 min |
| RHEL 9  | ~150 | ~30,000 | ~10 min |
| RHEL 8  | ~180 | ~35,000 | ~12 min |
| **Total** | **~410** | **~80,000** | **~27 min** |

Embedding times are approximate, based on the BGE-small model running on CPU with batch size 32.

## 11. Ingestion Pipeline

### 11.1 Pipeline Steps

```
python scripts/ingest.py --version 9

Steps:
  1. Discover guides for RHEL 9 from docs.redhat.com
  2. Download HTML-single pages (skip cached)
  3. Parse HTML → extract sections
  4. Chunk sections (150–800 tokens, section-aware)
  5. Embed chunks with BGE-small
  6. Upsert into LanceDB with version="9"
```

Each step should log its progress clearly:

```
[Step 1/6] Discovering guides for RHEL 9...
  Found 150 guides
[Step 2/6] Downloading HTML pages...
  [1/150] configuring_networking ... cached (skipped)
  [2/150] managing_storage ... downloading ... done (245 KB)
  ...
[Step 3/6] Parsing HTML...
  [1/150] configuring_networking → 48 sections
  ...
[Step 4/6] Chunking...
  Created 30,142 chunks from 150 guides
[Step 5/6] Embedding chunks...
  Batch [1/942] ... done
  Batch [2/942] ... done
  ...
[Step 6/6] Storing in LanceDB...
  Inserted 30,142 records into table 'rhel_docs'
  Total records in table: 30,142

Done! Ingested RHEL 9: 150 guides → 30,142 chunks
```

### 11.2 CLI Interface

```bash
# Ingest specific version
python scripts/ingest.py --version 9

# Ingest multiple versions
python scripts/ingest.py --version 8 --version 9 --version 10

# Ingest all versions
python scripts/ingest.py --all-versions

# Re-scrape (ignore cache, re-download HTML)
python scripts/ingest.py --version 10 --force-refresh

# Test with a small subset first (only first 10 guides)
python scripts/ingest.py --version 9 --limit 10

# List what's currently in the database
python scripts/ingest.py --list
# Output:
#   Version  Chunks
#   8        35,210
#   9        30,142
#   10       14,890
#   Total    80,242

# Delete a specific version from the database
python scripts/ingest.py --delete 8

# Drop everything and start fresh
python scripts/ingest.py --fresh --version 9
```

### 11.3 Chunking Strategy — Detailed

The chunker splits parsed sections into chunks suitable for embedding and retrieval. Key rules:

| Rule | Why |
|------|-----|
| **Target size: 500 tokens** | Large enough to contain meaningful context, small enough for precise retrieval |
| **Minimum size: 150 tokens** | Chunks smaller than this lack context. Merge with adjacent content instead. |
| **Maximum size: 800 tokens** | Chunks larger than this waste embedding space and dilute relevance. Split at paragraph boundaries. |
| **Never split code blocks** | A partial code block is useless. Keep code blocks whole, even if they exceed max size. |
| **Preserve section hierarchy** | Every chunk carries its heading path (e.g., "Ch 5 > 5.1 Networking > 5.1.2 DNS"). This gives the LLM context about where the chunk came from. |
| **Merge small sections** | If a section is only 50 tokens, merge it with the next section rather than creating a tiny chunk. |
| **Split at paragraph boundaries** | When a section is too large, split between paragraphs, not in the middle of a sentence. |

## 12. Search & Agent

### 12.1 Search Behavior

The search uses three techniques combined:

1. **Vector search** (cosine similarity) — Finds chunks whose meaning is similar to the question. Good for natural language questions like "how do I configure a static IP address?"
2. **BM25 full-text search** — Finds chunks containing the exact keywords. Good for specific terms like "nmcli connection modify", "SELinux boolean", or config file paths like `/etc/sysctl.conf`.
3. **RRF fusion** — Combines the two result lists into one, weighted 70% vector / 30% BM25 by default.

**Search parameters** (configurable in `config.py`):

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `search_top_k` | 20 | How many candidates each search method returns |
| `search_rerank_top_n` | 5 | How many final results to return after fusion |
| `rrf_k` | 60 | RRF smoothing constant (higher = more weight to lower-ranked results) |
| `vector_weight` | 0.7 | Weight for vector search in fusion (0.0–1.0) |

### 12.2 Agent Tools

The agent has two tools it can call to answer user questions:

```python
docs_search(query: str, version: str | None = None) -> str
    """
    Search RHEL documentation.

    Args:
        query: The search query (natural language or keywords)
        version: Optional RHEL version to filter by ("8", "9", or "10").
                 If None, searches all versions.

    Returns:
        Formatted string with top 5 matching chunks, including:
        - Guide title and section hierarchy
        - Relevance score
        - Chunk text (truncated)
        - Source URL
    """

docs_compare(query: str, versions: list[str]) -> str
    """
    Compare documentation across RHEL versions.

    Searches the same query across multiple versions and returns
    results grouped by version, so the agent can identify differences.

    Args:
        query: The topic to compare
        versions: List of versions to compare (e.g., ["8", "9", "10"])

    Returns:
        Formatted string with results grouped by version
    """
```

### 12.3 System Prompt

The agent's system prompt should establish it as a RHEL troubleshooting expert. Key focus areas:

- Package management (yum vs dnf differences across versions)
- Systemd service configuration and troubleshooting
- SELinux policy management and troubleshooting
- Networking (NetworkManager, nmcli, firewalld)
- Storage (LVM, Stratis, VDO, XFS, ext4)
- Security (crypto policies, FIPS mode, certificates, SSH hardening)
- Kernel tuning and performance (sysctl, tuned profiles)
- Container tools (podman, buildah, skopeo)
- Subscription management (subscription-manager, RHSM)
- Upgrade/migration paths (8→9, 9→10, leapp)
- Web console (cockpit)
- Ansible automation for RHEL

The system prompt should instruct the agent to:
- Always cite the guide title and URL in its answers
- Note when behavior differs between RHEL versions
- Warn about deprecated features or breaking changes
- Provide exact commands when available
- Suggest related topics the user might want to explore

## 13. Configuration

All settings are centralized in `config.py` using `pydantic-settings`. The implementer should create a `Settings` class with clear defaults:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `db_path` | `Path` | `data/lancedb` | Where LanceDB stores its files |
| `table_name` | `str` | `rhel_docs` | Name of the LanceDB table |
| `cache_dir` | `Path` | `data/html_cache` | Where downloaded HTML pages are cached |
| `embedding_model` | `str` | `BAAI/bge-small-en-v1.5` | HuggingFace model name for embeddings |
| `embedding_dim` | `int` | `384` | Vector dimensions (must match the model) |
| `embedding_batch_size` | `int` | `32` | How many chunks to embed at once |
| `chunk_min_tokens` | `int` | `150` | Minimum chunk size |
| `chunk_max_tokens` | `int` | `800` | Maximum chunk size |
| `chunk_target_tokens` | `int` | `500` | Ideal chunk size |
| `openrouter_api_key` | `str` | from `.env` | API key for the LLM |
| `search_top_k` | `int` | `20` | Candidates per search method |
| `search_rerank_top_n` | `int` | `5` | Final results after fusion |
| `rrf_k` | `int` | `60` | RRF smoothing constant |
| `vector_weight` | `float` | `0.7` | Vector vs BM25 weight |
| `scrape_delay` | `float` | `1.5` | Seconds between HTTP requests |
| `scrape_retries` | `int` | `3` | Retry count for failed downloads |
| `user_agent` | `str` | `rh-linux-docs-agent/1.0` | HTTP User-Agent header |
| `web_port` | `int` | `7933` | Port for the Gradio web UI |

## 14. Air-Gapped Deployment

### 14.1 Documentation Cache

For air-gapped (no internet) environments, the HTML cache must be pre-populated on a connected machine:

```bash
# === On a machine WITH internet ===

# Step 1: Scrape and cache all docs
python scripts/scrape.py --all-versions --cache-dir data/html_cache

# Step 2: Package the cache for transfer
tar czf rhel-docs-cache.tar.gz data/html_cache/

# Step 3: Transfer to air-gapped machine (USB, SCP, etc.)


# === On the AIR-GAPPED machine ===

# Step 4: Extract the cache
tar xzf rhel-docs-cache.tar.gz

# Step 5: Ingest from cache (no network needed for this step)
python scripts/ingest.py --all-versions --offline
```

### 14.2 Embedding Model

The embedding model (`BAAI/bge-small-en-v1.5`, ~130MB) is downloaded from HuggingFace Hub the first time the embedder runs. For air-gapped environments, pre-download it:

```bash
# === On a machine WITH internet ===

# Option A: Save model to a local directory
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
model.save('./models/bge-small')
print('Model saved to ./models/bge-small')
"
# Then update embedding_model in config.py to './models/bge-small'
# Transfer the models/ directory to the air-gapped machine

# Option B: Pre-populate the HuggingFace cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
# Copy ~/.cache/huggingface/ to the same path on the air-gapped machine

# Option C: Use a private HuggingFace mirror
export HF_ENDPOINT=https://your-artifactory.example.com/hf-mirror
```

### 14.3 Python Packages

Bundle all pip packages for offline install:

```bash
# === On a machine WITH internet ===

# Download all packages as wheel files
pip download -d ./wheels -e .

# Transfer the wheels/ directory to the air-gapped machine

# === On the AIR-GAPPED machine ===

# Install from local wheels (no internet needed)
pip install --no-index --find-links=./wheels -e .
```

### 14.4 LLM Access

The agent needs access to an LLM. In air-gapped environments:
- Use a locally-hosted model (e.g., vLLM, Ollama, or text-generation-inference)
- Or route through a private API endpoint
- Update the agent configuration to point to the local endpoint

## 15. Project Structure

```
rh-linux-docs-agent/
├── src/rh_linux_docs_agent/       # Main Python package (all source code lives here)
│   ├── __init__.py                # Makes this directory a Python package (can be empty)
│   ├── config.py                  # Settings class — all configuration in one place
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── discovery.py           # Finds all guide URLs for a RHEL version
│   │   └── fetcher.py             # Downloads + caches HTML pages
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── html_parser.py         # Extracts sections from HTML
│   │   └── models.py              # ParsedGuide, Section data classes
│   ├── chunker/
│   │   ├── __init__.py
│   │   ├── splitter.py            # Splits sections into chunks
│   │   └── models.py              # Chunk data class
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── embedder.py            # Loads model, converts text → vectors
│   │   ├── schema.py              # Defines LanceDB table columns
│   │   └── store.py               # Create/read/delete LanceDB records
│   ├── search/
│   │   ├── __init__.py
│   │   └── hybrid.py              # Vector + BM25 + RRF search
│   └── agent/
│       ├── __init__.py
│       ├── agent.py               # pydantic-ai agent definition
│       ├── app.py                 # Gradio web UI entry point
│       └── tools.py               # docs_search, docs_compare functions
├── scripts/
│   ├── scrape.py                  # CLI: download + cache HTML only
│   ├── ingest.py                  # CLI: full pipeline (scrape + parse + chunk + embed + store)
│   └── search_cli.py              # CLI: interactive search from terminal
├── data/                          # All generated data (not committed to git)
│   ├── html_cache/                # Cached HTML pages, organized by version
│   │   ├── 8/
│   │   ├── 9/
│   │   └── 10/
│   └── lancedb/                   # Vector database files
├── venv/                          # Python virtual environment (not committed to git)
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_search.py
├── pyproject.toml                 # Project config: name, version, dependencies
├── .env                           # API keys (not committed to git)
└── .gitignore                     # Lists files/dirs git should ignore: venv/, data/, .env
```

### 15.1 .gitignore

The `.gitignore` file must contain at minimum:

```gitignore
# Python virtual environment
venv/

# Generated data (too large for git)
data/

# Secrets
.env

# Python bytecode
__pycache__/
*.pyc

# IDE files
.idea/
.vscode/
```

## 16. Milestones

| # | Milestone | Description | Estimate | Details |
|---|-----------|-------------|----------|---------|
| 1 | **Project Setup** | Create project structure, pyproject.toml, venv, config | 0.5 day | Set up the repo, directory structure, dependencies, `.env`, `.gitignore`, and verify the venv works |
| 2 | **Scraper** | Guide discovery + HTML download + caching | 2 days | Build the discovery and fetcher modules, test against live docs.redhat.com, verify caching works |
| 3 | **HTML Parser** | Section extraction, code block handling, table conversion | 2 days | Build the HTML parser, handle all element types, test against several downloaded guides |
| 4 | **Chunker** | Section-aware splitting with token counting | 1 day | Build the splitter, test chunk sizes, verify code blocks aren't split |
| 5 | **Indexer** | Embedding + LanceDB storage | 1 day | Build the embedder and store modules, verify vectors are correct dimension, test insert/delete |
| 6 | **Search** | Hybrid vector + BM25 + RRF | 1 day | Build the search module, test with sample queries, tune weights |
| 7 | **Agent & Web UI** | pydantic-ai agent + Gradio chat | 1 day | Build the agent with tools, system prompt, and web UI |
| 8 | **Multi-version** | Version filtering, cross-version comparison | 0.5 day | Add version filter to search, build docs_compare tool |
| 9 | **API Integration** | Expose search as API for external tool consumption | 0.5 day | REST or tool-based API wrapping the search functionality |
| 10 | **Air-gapped** | Offline scrape cache, model bundling | 1 day | Test full offline workflow: cached HTML + bundled model + local LLM |
| 11 | **Testing & Tuning** | Search quality evaluation, chunk size tuning | 2 days | Create test queries, evaluate result quality, adjust parameters |

**Total estimate: ~12.5 days**

## 17. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HTML structure changes on docs.redhat.com | Parser breaks, ingestion fails | Version the parser, add structure validation, log parse warnings, alert on unexpected HTML |
| Rate limiting / blocking by docs.redhat.com | Can't download docs | Aggressive caching (download once), respect robots.txt, use reasonable delays, set proper User-Agent |
| Large corpus (~80K chunks) slows search | High query latency | LanceDB handles this well; add IVF index if needed; consider reducing chunk count by increasing target size |
| HTML tables lose structure in chunking | Poor search results for tabular data | Convert tables to markdown format before chunking; keep small tables as single chunks |
| Code blocks split across chunks | Broken code in search results | Chunker treats code blocks as atomic — never splits them, even if they exceed max chunk size |
| docs.redhat.com requires authentication for some content | Missing documentation | Log and skip authenticated pages; document which guides are subscription-only |
| Embedding model produces poor results for RHEL-specific terms | Low search relevance | BGE-small handles technical English well; if needed, explore domain-specific models or fine-tuning |

## 18. Testing Strategy

### 18.1 Unit Tests

Each module should have basic unit tests:
- **Parser** — Feed it sample HTML, verify sections are extracted correctly
- **Chunker** — Feed it parsed sections, verify chunk sizes are within bounds and code blocks aren't split
- **Search** — Feed it a known database, verify relevant results appear in top 5

### 18.2 Integration Tests

- **End-to-end** — Scrape 2–3 small guides, ingest them, query, verify the agent returns relevant answers
- **Version comparison** — Ingest same topic from RHEL 8 and 9, verify `docs_compare` shows differences

### 18.3 Search Quality Evaluation

Create a set of 20–30 test queries with expected answers:
- "How to configure a static IP address on RHEL 9?"
- "What are the differences in firewalld between RHEL 8 and 9?"
- "How to enable FIPS mode?"
- "How to migrate from yum to dnf?"

Manually evaluate: does the agent return correct, well-cited answers?

## 19. Future Enhancements

- **Fedora docs** — Add Fedora documentation (public AsciiDoc source on GitHub)
- **CentOS Stream docs** — Upstream RHEL content
- **Knowledge articles** — Integrate Red Hat Knowledgebase articles (requires subscription API)
- **Release notes diffing** — Automated comparison of release notes between versions
- **Subscription-gated docs** — Support authenticated scraping for subscriber-only content
- **Incremental updates** — Detect when docs.redhat.com content changes and re-ingest only updated guides
- **Search analytics** — Log queries and results to identify gaps in documentation coverage
