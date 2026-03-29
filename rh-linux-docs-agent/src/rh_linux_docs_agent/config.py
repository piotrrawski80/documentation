"""
config.py — Central configuration for the RHEL docs agent.

All settings live here. Change defaults here, or override them with environment
variables or a .env file in the project root.

Usage:
    from rh_linux_docs_agent.config import settings
    print(settings.db_path)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables and .env file.

    pydantic-settings automatically reads a .env file and populates these fields.
    Any field can be overridden by setting an environment variable with the same
    name (uppercase). Example: DB_PATH=/tmp/lancedb python scripts/ingest.py
    """

    model_config = SettingsConfigDict(
        # Look for .env in the project root (two levels up from this file)
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        # Don't crash if .env doesn't exist — just use defaults
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    # Parent directory for per-version LanceDB databases.
    # Each version gets its own subdirectory: data/lancedb_v2/rhel9/, etc.
    db_path: Path = Path("data/lancedb_v2")

    # Legacy single-DB path (used as fallback if versioned path doesn't exist).
    db_path_legacy: Path = Path("data/lancedb")

    # The name of the table inside LanceDB that holds all document chunks.
    table_name: str = "rhel_docs"

    # ── HTML Cache ────────────────────────────────────────────────────────────
    # Where downloaded HTML pages are saved. Once cached, they're never
    # re-downloaded unless you use --force-refresh.
    cache_dir: Path = Path("data/html_cache")

    # ── Embedding Model ───────────────────────────────────────────────────────
    # The HuggingFace model used to convert text into vectors.
    # BGE-small-en-v1.5 is ~130MB, runs on CPU, and produces 384-dim vectors.
    # For air-gapped use, set this to a local path like "./models/bge-small"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Must match the model's output size. BGE-small produces 384-dim vectors.
    # If you change the model, you MUST change this too.
    embedding_dim: int = 384

    # How many chunks to embed at once. Larger = faster, but uses more RAM.
    # Reduce to 8 or 16 if you run out of memory.
    embedding_batch_size: int = 32

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Minimum chunk size in tokens. Chunks smaller than this are too small
    # to be useful and will be merged with adjacent content.
    chunk_min_tokens: int = 150

    # Maximum chunk size in tokens. Chunks larger than this waste embedding
    # capacity. Exception: code blocks are never split even if they exceed this.
    chunk_max_tokens: int = 800

    # Target chunk size. The splitter aims for chunks close to this size.
    chunk_target_tokens: int = 500

    # ── LLM (via OpenRouter) ──────────────────────────────────────────────────
    # Your OpenRouter API key. Get one at https://openrouter.ai
    # Must be set in .env file: OPENROUTER_API_KEY=sk-or-...
    openrouter_api_key: str = ""

    # The LLM model to use for generating answers. This runs on OpenRouter.
    # deepseek-chat is high-quality and very affordable.
    llm_model: str = "deepseek/deepseek-chat-v3-0324"

    # ── Search ────────────────────────────────────────────────────────────────
    # How many candidate results each search method (vector + BM25) returns
    # before merging. More candidates = better recall but slower.
    search_top_k: int = 20

    # How many final results to return after merging vector + BM25 results.
    # These are the chunks passed to the LLM as context.
    # 4 keeps context focused; the score threshold below may drop it further.
    search_rerank_top_n: int = 4

    # Minimum cross-encoder rerank score to include a chunk in context.
    # Chunks scoring below this are dropped even if top_n is not reached.
    # Cross-encoder range is roughly [-10, 12]; 3.0 filters weak matches.
    # Set to -999.0 to disable the threshold entirely.
    rerank_score_threshold: float = 3.0

    # Maximum text-overlap ratio (0-1) between two chunks before the
    # lower-scored one is considered a duplicate and removed.
    dedup_similarity_threshold: float = 0.70

    # RRF (Reciprocal Rank Fusion) smoothing constant.
    # Higher values give more weight to lower-ranked results.
    # 60 is the standard default from the RRF paper.
    rrf_k: int = 60

    # How much weight to give vector search vs BM25 in result fusion.
    # 0.7 = 70% vector (semantic), 30% BM25 (keyword).
    # Increase toward 1.0 for more semantic matching.
    # Decrease toward 0.0 for more exact keyword matching.
    vector_weight: float = 0.7

    # ── Scraping ──────────────────────────────────────────────────────────────
    # Seconds to wait between HTTP requests to docs.redhat.com.
    # Be polite — don't set this below 1.0 or you may get blocked.
    scrape_delay: float = 1.5

    # How many times to retry a failed HTTP request before giving up.
    scrape_retries: int = 3

    # HTTP User-Agent header sent with every request.
    # A descriptive User-Agent is good practice and helps site operators
    # understand who is accessing their site.
    user_agent: str = "rh-linux-docs-agent/1.0 (documentation indexer; https://github.com/example/rh-linux-docs-agent)"

    # ── Web UI ────────────────────────────────────────────────────────────────
    # Port for the Gradio web chat interface.
    # Open http://localhost:7933 after starting the agent.
    web_port: int = 7933

    # ── RHEL Versions ─────────────────────────────────────────────────────────
    # Supported RHEL versions, in order of priority.
    supported_versions: list[str] = ["10", "9", "8"]

    # Base URL for RHEL documentation.
    docs_base_url: str = "https://docs.redhat.com/en/documentation/red_hat_enterprise_linux"


    # ── Version-aware paths ──────────────────────────────────────────────────

    def db_path_for_version(self, version: str) -> Path:
        """
        Return the LanceDB directory for a specific RHEL version.

        Each version has its own directory: data/lancedb_v2/rhel9/, etc.
        If the versioned directory doesn't exist yet, returns the path anyway
        (it will be created on first ingest). No legacy fallback — each
        version's data is strictly isolated.

        Args:
            version: RHEL major version string ("8", "9", or "10").

        Returns:
            Path to the LanceDB directory for this version.
        """
        return self.db_path / f"rhel{version}"

    def parsed_dir_for_version(self, version: str) -> Path:
        """Return parsed JSON directory for a version: data/parsed/rhel{v}/"""
        return Path("data/parsed") / f"rhel{version}"

    def chunked_dir_for_version(self, version: str) -> Path:
        """Return chunked JSON directory for a version: data/chunked/rhel{v}/"""
        return Path("data/chunked") / f"rhel{version}"


# Global settings instance — import this in other modules.
# All modules should do: from rh_linux_docs_agent.config import settings
settings = Settings()
