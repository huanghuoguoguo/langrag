"""Base embedder interface."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation.

    Embedders convert text strings into vector representations.

    This interface supports both sync and async implementations:
    - Override `embed()` for sync implementations (local models)
    - Override `embed_async()` for async implementations (remote APIs)

    The default `embed_async()` wraps `embed()` for backward compatibility.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (sync version).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If texts is empty
        """
        pass

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (async version).

        Override this method for async implementations (e.g., remote API calls).
        Default implementation wraps the sync `embed()` method.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If texts is empty
        """
        import asyncio
        return await asyncio.to_thread(self.embed, texts)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Size of embedding vectors produced by this embedder
        """
        pass
