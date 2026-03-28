# Runbook

Operational procedures for the RHEL 9 Documentation Agent. All commands assume the virtual environment is activated and the working directory is the project root.

---

## 1. First-time setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -e .
```

Create `.env` in the project root (optional, needed only for LLM answer generation):

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

---

## 2. Full ingestion pipeline

The full pipeline runs all stages sequentially: discover guides, download HTML, parse, chunk, embed, and store.

```bash
python scripts/ingest.py --version 9
```

Options:
- `--limit 10` — process only the first 10 guides (useful for testing)
- `--force-refresh` — re-download HTML even if cached

**Expected output:** ~140 guides scraped, ~28,000 chunks embedded and indexed. Takes 15-30 minutes on CPU depending on hardware.

### Running stages individually

If you need to re-run a specific stage without repeating the full pipeline:

```bash
# Stage 1: Scrape HTML only (download + cache)
python scripts/scrape.py --version 9

# Stage 2: Parse cached HTML into section JSON
python scripts/parse_docs.py --docs-dir data/html_cache/rhel9

# Stage 3: Chunk sections
python scripts/chunk_docs.py --version 9

# Stage 4: Embed + index into LanceDB
python scripts/index_docs.py --version 9
```

To re-index a single guide without touching others:

```bash
python scripts/index_docs.py --version 9 --guide configuring_and_managing_networking
```

To drop and rebuild the entire index:

```bash
python scripts/index_docs.py --version 9 --fresh
```

---

## 3. Verify the index

Check database statistics after indexing:

```bash
python scripts/index_docs.py --version 9 --stats
```

Expected output includes: total chunks, versions present, doc_type distribution, guide count.

---

## 4. Search and QA

### Interactive search

```bash
python scripts/search_cli.py
```

Type queries interactively. Supports `--version 9` filter and `--versions 8,9` for cross-version (when available).

### QA demo

Run the 6 built-in demo queries:

```bash
python scripts/qa_demo.py
```

Run a custom query:

```bash
python scripts/qa_demo.py --query "How to configure firewalld to allow SSH and HTTPS?"
```

Show retrieved context without LLM generation:

```bash
python scripts/qa_demo.py --context-only
```

Skip the reranker (faster, lower quality):

```bash
python scripts/qa_demo.py --no-rerank
```

### QA without API key

The system works without an OpenRouter API key. When no key is set, it produces structured answers showing verbatim retrieved context with citations. No commands or procedures are synthesized.

```bash
python scripts/qa_showcase.py
```

---

## 5. Evaluate retrieval quality

Run the 20-query evaluation set:

```bash
python scripts/eval_retrieval.py
```

Compare retrieval modes:

```bash
# Hybrid + reranker (default, best quality)
python scripts/eval_retrieval.py

# Hybrid without reranker
python scripts/eval_retrieval.py --no-rerank

# Vector-only
python scripts/eval_retrieval.py --mode vector

# BM25-only
python scripts/eval_retrieval.py --mode bm25
```

Run a single query from the evaluation set:

```bash
python scripts/eval_retrieval.py --query 3
```

---

## 6. Updating the documentation index

When Red Hat publishes new documentation:

```bash
# Re-scrape (downloads new/changed pages)
python scripts/scrape.py --version 9 --force-refresh

# Re-run the full pipeline
python scripts/ingest.py --version 9
```

To update a single guide:

```bash
# Delete the cached HTML and re-download
rm data/html_cache/rhel9/configuring_and_managing_networking.html
python scripts/ingest.py --version 9 --limit 1
# Or re-index just that guide
python scripts/index_docs.py --version 9 --guide configuring_and_managing_networking --fresh
```

---

## 7. Configuration changes

All configuration lives in `src/rh_linux_docs_agent/config.py`. Override any setting via environment variable or `.env`:

```bash
# Change the LLM model
echo "LLM_MODEL=anthropic/claude-3.5-sonnet" >> .env

# Change search parameters
echo "SEARCH_RERANK_TOP_N=3" >> .env
echo "RERANK_SCORE_THRESHOLD=4.0" >> .env

# Use a local embedding model path (for air-gapped)
echo "EMBEDDING_MODEL=./models/bge-small" >> .env
```

---

## 8. Air-gapped deployment

### On a connected machine

```bash
# 1. Download all HTML
python scripts/scrape.py --version 9

# 2. Download embedding model
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-small-en-v1.5'); m.save('./models/bge-small')"

# 3. Download reranker model
python -c "from sentence_transformers import CrossEncoder; m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); m.save('./models/reranker')"

# 4. Bundle Python packages
pip download -d ./wheels -e .

# 5. Package everything
tar czf rhel-docs-agent-bundle.tar.gz \
    data/html_cache/ models/ wheels/ \
    src/ scripts/ pyproject.toml .gitignore
```

### On the air-gapped machine

```bash
tar xzf rhel-docs-agent-bundle.tar.gz
python -m venv venv && source venv/bin/activate
pip install --no-index --find-links=./wheels -e .

# Update config to use local models
echo "EMBEDDING_MODEL=./models/bge-small" > .env

# Run the ingestion pipeline (no network needed)
python scripts/ingest.py --version 9

# Query (offline mode — no LLM, shows retrieved context directly)
python scripts/qa_demo.py --query "How to configure SELinux?"
```

---

## 9. Troubleshooting

**"OPENROUTER_API_KEY not set"** — Expected when no `.env` file exists. The system falls back to offline mode and shows retrieved context directly. To enable LLM generation, create a `.env` file with your key.

**"No relevant documentation found"** — The LanceDB index may be empty or corrupted. Run `python scripts/index_docs.py --version 9 --stats` to check. If empty, re-run the ingestion pipeline.

**Slow first query** — The embedding model (~130MB) and reranker (~80MB) are loaded on the first query. Subsequent queries reuse the loaded models. First query takes ~5-10s, later queries ~1-2s.

**PyTorch DLL error on Windows** — If you see `WinError 1114` or `c10.dll` errors, reinstall PyTorch CPU:
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

**BM25 search fails** — The FTS index may not be built. Run:
```bash
python scripts/index_docs.py --version 9 --build-index
```
The system falls back to vector-only search if BM25 is unavailable.
