"""
Embedder implementations for the Web layer.

This module provides concrete Embedder implementations that can be injected into
the LangRAG core. These embedders demonstrate how to integrate external embedding
services (like OpenAI API) with the LangRAG framework.

Design Decisions:
-----------------
1. **External Injection Pattern**: Embedders are created and managed by the web layer,
   then injected into LangRAG core components. This allows the web application to
   control API keys, endpoints, and lifecycle.

2. **BaseEmbedder Interface**: All embedders implement LangRAG's BaseEmbedder interface,
   ensuring compatibility with core components like IndexProcessor and VectorStore.

3. **Lazy Initialization**: SeekDBEmbedder uses lazy initialization to avoid loading
   the model until actually needed, reducing startup time.

Example Usage:
--------------
    # OpenAI-compatible embedder
    embedder = WebOpenAIEmbedder(
        base_url="https://api.openai.com/v1",
        api_key="sk-xxx",
        model="text-embedding-3-small"
    )

    # SeekDB built-in embedder (uses all-MiniLM-L6-v2)
    embedder = SeekDBEmbedder()

    # Use with LangRAG components
    vectors = embedder.embed(["Hello world", "How are you?"])
"""

import logging

import httpx

from langrag.llm.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)


class WebOpenAIEmbedder(BaseEmbedder):
    """
    OpenAI-compatible Embedder implementation.

    This embedder works with any API that follows the OpenAI embeddings format,
    including OpenAI, Azure OpenAI, and local models served via compatible APIs
    (e.g., LocalAI, Ollama with OpenAI compatibility layer).

    Attributes:
        base_url: The API base URL (e.g., "https://api.openai.com/v1")
        api_key: API authentication key
        model: Model identifier (e.g., "text-embedding-3-small")
        batch_size: Maximum texts per API call (default: 100)

    Why httpx instead of openai SDK:
        We use httpx directly for synchronous embedding calls because:
        1. LangRAG core components are currently synchronous
        2. Avoids event loop conflicts when called from sync context
        3. Simpler error handling and timeout configuration
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 100
    ):
        """
        Initialize the OpenAI-compatible embedder.

        Args:
            base_url: API endpoint base URL (trailing slash will be stripped)
            api_key: Authentication key for the API
            model: Model name to use for embeddings
            batch_size: Maximum texts per API call (default: 100).
                       Larger values are more efficient but may hit API limits.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=60.0)
        self._dimension = 1536  # Default for text-embedding-3-small, updated on first call

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        For large input lists, texts are processed in batches to avoid
        hitting API rate limits or request size limits.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            Exception: If the API call fails (no fallback, fail fast)
        """
        if not texts:
            return []

        # Process in batches if needed
        if len(texts) <= self.batch_size:
            return self._embed_single_batch(texts)

        # Batch processing for large inputs
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Processing {len(texts)} texts in {total_batches} batches "
            f"(batch_size={self.batch_size})"
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            batch_embeddings = self._embed_single_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single batch of texts.

        Args:
            texts: List of strings to embed (should be <= batch_size)

        Returns:
            List of embedding vectors

        Raises:
            Exception: If the API call fails
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": self.model
        }

        try:
            resp = self.client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Sort by index to ensure correct order
            results = data.get("data", [])
            results.sort(key=lambda x: x.get("index", 0))

            vector_list = [item["embedding"] for item in results]

            # Update dimension based on actual response
            if vector_list:
                self._dimension = len(vector_list[0])

            return vector_list

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Fail fast - don't use fallback, let caller handle the error
            raise

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class SeekDBEmbedder(BaseEmbedder):
    """
    SeekDB built-in Embedder using all-MiniLM-L6-v2.

    This embedder uses pyseekdb's default embedding function, which provides
    a lightweight, local embedding solution without requiring external API calls.

    Characteristics:
        - Model: all-MiniLM-L6-v2 (sentence-transformers)
        - Dimension: 384
        - Language: Multilingual support
        - Speed: Fast inference on CPU

    Why use this embedder:
        1. No API costs - runs entirely locally
        2. No network latency - faster for batch processing
        3. Privacy - data never leaves your machine
        4. Offline capable - works without internet

    Trade-offs:
        - Lower quality than large models (e.g., text-embedding-3-large)
        - Fixed dimension (384 vs 1536+ for OpenAI)
        - Requires pyseekdb installation
    """

    def __init__(self):
        """Initialize with lazy loading of the embedding function."""
        self._dimension = 384  # all-MiniLM-L6-v2 dimension
        self._embedding_function = None

    def _initialize(self):
        """
        Lazily initialize the embedding function.

        We delay initialization because:
        1. Loading sentence-transformers model takes time
        2. The embedder might be created but never used
        3. Reduces application startup time
        """
        if self._embedding_function is None:
            try:
                import pyseekdb
                self._embedding_function = pyseekdb.get_default_embedding_function()
                logger.info("SeekDB default embedding function initialized (all-MiniLM-L6-v2)")
            except ImportError:
                logger.error("pyseekdb not installed. Install with: pip install pyseekdb")
                raise ImportError("pyseekdb is required for SeekDB embedder")
            except Exception as e:
                logger.error(f"Failed to initialize SeekDB embedding function: {e}")
                raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using pyseekdb's default function.

        Processes texts in batches to balance memory usage and performance.
        Batch size of 32 is chosen as a reasonable default for CPU inference.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (384-dimensional)
        """
        try:
            self._initialize()

            # Process in batches for better memory management
            batch_size = 32
            all_embeddings = []

            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(texts)} texts in {total_batches} batches (batch_size={batch_size})")

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
                batch_embeddings = self._embedding_function(batch)
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Completed embedding {len(texts)} texts")
            return all_embeddings

        except Exception as e:
            logger.error(f"SeekDB embedding failed: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self._dimension
