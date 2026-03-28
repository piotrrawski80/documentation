"""
indexer/embedder.py — Converts text chunks into numerical vectors (embeddings).

An embedding is a list of numbers (384 numbers for our model) that represents
the meaning of a text. Texts with similar meanings produce similar numbers,
which is how semantic search works.

We use BAAI/bge-small-en-v1.5:
- Small: ~130MB model, runs on CPU without a GPU
- Fast: ~3,000 chunks/minute on CPU with batch_size=32
- Good quality: strong performance on technical English text
- Local: runs entirely on your machine, no API calls needed
- Air-gap friendly: can be pre-downloaded and bundled offline

How semantic search works:
1. At index time: embed each chunk's text → store the vector in LanceDB
2. At query time: embed the user's question → find stored vectors closest to it
   (using cosine similarity — measures the "angle" between two vectors)

Usage:
    embedder = Embedder()
    texts = ["How to configure static IP", "NetworkManager configuration guide"]
    vectors = embedder.embed(texts)
    # vectors is a list of 384-element float lists, one per text
"""

import logging
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper around the sentence-transformers embedding model.

    Loads the BGE-small model once and reuses it for all embedding calls.
    Processes chunks in batches for efficiency.

    The model is downloaded from HuggingFace Hub the first time it's used
    (~130MB). Subsequent runs use the cached model in ~/.cache/huggingface/.

    For air-gapped environments:
        Pre-download with: python -c "from sentence_transformers import
        SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
        Then copy ~/.cache/huggingface/ to the air-gapped machine.

    Attributes:
        model: The loaded SentenceTransformer model instance.

    Example:
        embedder = Embedder()
        vectors = embedder.embed(["text one", "text two"])
        assert len(vectors[0]) == 384  # BGE-small produces 384-dim vectors
    """

    def __init__(self) -> None:
        """Load the embedding model. This may take a few seconds on first run."""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        # Import here to avoid loading ~130MB model until it's actually needed
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into embedding vectors.

        Processes texts in batches to balance speed and memory usage.
        Each text is converted to a 384-dimensional float vector.

        Args:
            texts: List of text strings to embed. Can be any length.

        Returns:
            List of vectors, one per input text. Each vector is a list of
            384 float values. The i-th vector corresponds to texts[i].

        Example:
            vectors = embedder.embed(["configure static IP", "nmcli setup"])
            # vectors[0] and vectors[1] will be similar (both about networking)
        """
        if not texts:
            return []

        # sentence-transformers handles batching internally, but we add
        # BGE-specific preprocessing (adding instruction prefix for queries)
        vectors = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )

        # Convert numpy arrays to plain Python lists (required for LanceDB/JSON)
        return [v.tolist() for v in vectors]

    def embed_with_progress(
        self,
        texts: list[str],
        description: str = "Embedding",
    ) -> list[list[float]]:
        """
        Embed texts with a rich progress bar shown in the terminal.

        Use this for large batches (like during ingestion) where you want to
        see how fast embedding is progressing.

        Args:
            texts: List of texts to embed.
            description: Label shown on the progress bar.

        Returns:
            List of embedding vectors, same as embed().
        """
        if not texts:
            return []

        batch_size = settings.embedding_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        all_vectors: list[list[float]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} batches"),
        ) as progress:
            task = progress.add_task("", total=total_batches)

            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                vectors = self.embed(batch)
                all_vectors.extend(vectors)
                progress.advance(task)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for search.

        BGE models recommend a specific instruction prefix for queries
        (not for documents) to improve retrieval quality.

        Args:
            query: The user's search query.

        Returns:
            Single embedding vector (list of 384 floats).
        """
        # BGE models use an instruction prefix for queries to improve retrieval.
        # The prefix tells the model "this is a search query, not a passage".
        instruction_prefix = "Represent this sentence for searching relevant passages: "
        prefixed_query = f"{instruction_prefix}{query}"

        vectors = self.embed([prefixed_query])
        return vectors[0]
