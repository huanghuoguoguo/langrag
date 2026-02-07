"""Base embedder interface."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation.

    Embedders convert text strings into vector representations.

    This interface supports both sync and async implementations:
    - Override `embed()` for sync implementations (local models)
    - Override `embed_async()` for async implementations (remote APIs, IPC)

    For async-only implementations (e.g., plugin adapters that only have async RPC),
    override `embed_async()` and leave `embed()` to raise NotImplementedError.

    The default `embed_async()` wraps `embed()` for backward compatibility.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (sync version).

        Override this method for sync implementations.
        Default implementation raises NotImplementedError to support async-only embedders.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            NotImplementedError: If only async implementation is available
            ValueError: If texts is empty
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sync embed(). "
            "Use embed_async() instead."
        )

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (async version).

        Override this method for async implementations (e.g., remote API calls, IPC).
        Default implementation wraps the sync `embed()` method.

        For async-only implementations, override this method and leave `embed()`
        to use the default NotImplementedError.

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
