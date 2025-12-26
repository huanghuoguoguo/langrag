"""SeekDB embedder using built-in embedding function.

SeekDB provides a default embedding function (all-MiniLM-L6-v2)
that can be used independently of SeekDB's vector storage.
"""

from typing import List
from loguru import logger

try:
    import pyseekdb
    SEEKDB_AVAILABLE = True
except ImportError:
    SEEKDB_AVAILABLE = False
    logger.debug("pyseekdb not installed - SeekDBEmbedder unavailable")

from ..base import BaseEmbedder


class SeekDBEmbedder(BaseEmbedder):
    """Embedder using SeekDB's built-in embedding function.

    Uses pyseekdb's default embedding model (all-MiniLM-L6-v2)
    which produces 384-dimensional embeddings. This embedder is
    completely decoupled from SeekDB's vector storage - you can
    use SeekDB embeddings with any vector store.

    The embedding model runs locally and doesn't require API calls.

    Attributes:
        _embedding_function: SeekDB's embedding function
        _dimension: Embedding dimension (384 for all-MiniLM-L6-v2)

    Example:
        >>> embedder = SeekDBEmbedder()
        >>> vectors = embedder.embed(["Hello world", "Test text"])
        >>> len(vectors[0])  # Dimension
        384
    """

    def __init__(self):
        """Initialize SeekDB embedder.

        Raises:
            ImportError: If pyseekdb is not installed
        """
        if not SEEKDB_AVAILABLE:
            raise ImportError(
                "pyseekdb is required for SeekDBEmbedder. "
                "Install with: pip install pyseekdb"
            )

        # Get SeekDB's default embedding function
        self._embedding_function = pyseekdb.get_default_embedding_function()
        self._dimension = 384  # all-MiniLM-L6-v2 dimension

        logger.info(
            "Initialized SeekDBEmbedder with all-MiniLM-L6-v2 model "
            "(dimension=384, local inference)"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SeekDB's embedding function.

        Args:
            texts: List of text strings to embed

        Returns:
            List of 384-dimensional embedding vectors

        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        logger.debug(f"Generating {len(texts)} embeddings with SeekDB")

        try:
            # SeekDB's embedding function handles batches
            embeddings = self._embedding_function(texts)

            logger.debug(
                f"Generated {len(embeddings)} embeddings "
                f"(dim={len(embeddings[0]) if embeddings else 0})"
            )

            return embeddings

        except Exception as e:
            logger.error(f"SeekDB embedding failed: {e}")
            raise RuntimeError(f"SeekDB embedding generation failed: {e}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            384 (dimension of all-MiniLM-L6-v2 model)
        """
        return self._dimension
