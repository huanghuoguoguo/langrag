"""Base embedder interface."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation.

    Embedders convert text strings into vector representations.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If texts is empty
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Size of embedding vectors produced by this embedder
        """
        pass
