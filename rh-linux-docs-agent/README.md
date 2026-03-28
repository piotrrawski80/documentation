# RHEL Documentation Agent

A RAG (Retrieval-Augmented Generation) system for Red Hat Enterprise Linux 9 documentation. Retrieves relevant documentation chunks via hybrid search, reranks them with a cross-encoder, and generates grounded answers with mandatory citations.

**Current scope:** RHEL 9 only (~140 guides, ~28,000 chunks indexed).

## How it works

```
docs.redhat.com HTML  →  Parser  →  Chunker  →  Embedder  →  LanceDB
                                                                  ↓
                              Answer  ←  LLM  ←  Context  ←  Retriever
                         (with citations)        assembly    (hybrid + rerank
                                                              + dedup)
```

1. HTML pages are scraped from `docs.redhat.com` and cached locally
2. A parser extracts structured sections (headings, code blocks, tables, admonitions)
3. A chunker splits sections into 100-800 token pieces, keeping code blocks intact
4. BGE-small-en-v1.5 converts each chunk into a 384-dim vector
5. LanceDB stores chunks with 22 metadata columns + vectors
6. At query time: hybrid search (vector + BM25 + RRF) → cross-encoder reranking → score threshold → deduplication → context assembly → LLM generation with strict grounding

## Requirements

- Python >= 3.13
- ~500MB disk for models (embedding 130MB + reranker 80MB, downloaded on first run)
- ~1GB disk for LanceDB index
- OpenRouter API key (optional — system works offline without it)

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install the project and all dependencies
pip install -e .

# (Optional) Set up LLM access for answer generation
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

## Quick start

If the index is already built (data/lancedb/ exists with the `rhel_docs` table):

```bash
# Run a single QA query
python scripts/qa_demo.py --query "How to configure a static IP on RHEL 9?"

# Run the 6 built-in demo queries
python scripts/qa_demo.py

# Show retrieved context without calling the LLM
python scripts/qa_demo.py --context-only

# Run the 20-query retrieval evaluation
python scripts/eval_retrieval.py
```

If building from scratch, see the [Runbook](RUNBOOK.md) for the full ingestion pipeline.

## Project structure

```
src/rh_linux_docs_agent/
    config.py               Central configuration (pydantic-settings)
    scraper/
        discovery.py        Guide URL discovery from landing pages
        fetcher.py          HTML download + disk caching
        parser.py           Recursive section extraction from HTML
    chunker/
        splitter.py         Section-aware, token-bounded chunking
    indexer/
        embedder.py         BGE-small embedding wrapper
        schema.py           PyArrow schema + record converters
        store.py            LanceDB table operations + index building
    search/
        hybrid.py           Vector + BM25 + RRF fusion
        reranker.py         Cross-encoder reranker
        retriever.py        Full pipeline: classify → search → rerank → dedup
    agent/
        qa.py               QA engine: context assembly + LLM + citations

scripts/
    ingest.py               Full pipeline: scrape → parse → chunk → embed → store
    scrape.py               Download HTML only
    parse_docs.py           Parse cached HTML into section JSON
    chunk_docs.py           Chunk sections into embedding-ready pieces
    index_docs.py           Embed + store chunks in LanceDB
    search_demo.py          5 example search queries
    search_cli.py           Interactive terminal search
    eval_retrieval.py       20-query retrieval evaluation
    qa_demo.py              QA demo with rich output
    qa_showcase.py          QA showcase (offline mode)

data/                       (not in git)
    html_cache/rhel9/       Cached HTML pages
    parsed/rhel9/           Parsed section JSON
    chunked/rhel9/          Chunked JSON
    lancedb/                Vector database files
```

## Key configuration

All settings live in `src/rh_linux_docs_agent/config.py` and can be overridden via `.env` or environment variables.

| Setting | Default | Purpose |
|---------|---------|---------|
| `db_path` | `data/lancedb` | LanceDB storage directory |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `llm_model` | `deepseek/deepseek-chat-v3-0324` | LLM for answer generation |
| `search_top_k` | `20` | Retrieval candidates |
| `search_rerank_top_n` | `4` | Final chunks after reranking |
| `rerank_score_threshold` | `3.0` | Minimum rerank score to keep |
| `vector_weight` | `0.7` | Vector vs BM25 weight in RRF |

## Grounding policy

The system answers **only** from retrieved evidence. It will not introduce commands, paths, or procedures from its training data. When evidence is incomplete, it says so explicitly and links to the full documentation. Every claim is cited with guide title, section heading, and URL.

See [PRD-RH-Linux-v2.md](PRD-RH-Linux-v2.md) for the full grounding rules and architecture details.

## Retrieval quality

Measured on 20 test queries across 7 categories (networking, security, storage, containers, system management, identity, virtualization):

| Metric | Score |
|--------|-------|
| Hit@1 | 85% |
| Hit@3 | 95% |
| Hit@5 | 100% |
| MRR | 0.917 |

## License

Internal project. Not for public distribution.
