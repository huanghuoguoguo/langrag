"""Mock embedder for testing (no external API)."""

import random
from loguru import logger

from ..base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Generates deterministic random embeddings for testing.

    WARNING: This embedder is NOT suitable for production use.
    It generates random vectors for testing purposes only.

    Attributes:
        dimension: Embedding vector dimension
        seed: Random seed for reproducibility
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        """Initialize the mock embedder.

        Args:
            dimension: Size of embedding vectors
            seed: Random seed for deterministic output
        """
        self._dimension = dimension
        self.seed = seed
        random.seed(seed)
        logger.warning(
            "Using MockEmbedder - NOT for production use! "
            "Replace with real embedder for actual applications."
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for texts.

        Uses text hash for deterministic randomness per unique text.

        Args:
            texts: List of text strings to embed

        Returns:
            List of normalized random vectors

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        logger.debug(f"Generating {len(texts)} mock embeddings")

        embeddings = []
        for text in texts:
            # Use text hash for deterministic randomness
            text_seed = hash(text) + self.seed
            rng = random.Random(text_seed)

            # Generate random vector
            vec = [rng.gauss(0, 1) for _ in range(self._dimension)]

            # Normalize to unit length
            magnitude = sum(x**2 for x in vec) ** 0.5
            if magnitude > 0:
                vec = [x / magnitude for x in vec]
            else:
                vec = [0.0] * self._dimension

            embeddings.append(vec)

        return embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
