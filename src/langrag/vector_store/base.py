"""Base vector store interface."""

from abc import ABC, abstractmethod
from typing import Optional
from ..core.chunk import Chunk
from ..core.search_result import SearchResult
from .capabilities import VectorStoreCapabilities, SearchMode


class BaseVectorStore(ABC):
    """Abstract base class for vector storage and similarity search.

    Vector stores persist chunks with their embeddings and provide
    efficient similarity-based retrieval. They may support different
    search modes: vector similarity, full-text keyword search, or
    hybrid search combining both.
    """

    @property
    @abstractmethod
    def capabilities(self) -> VectorStoreCapabilities:
        """Get the search capabilities supported by this store.

        Returns:
            Capability declaration for this vector store
        """
        pass

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store.

        Args:
            chunks: Chunks with embeddings to store

        Raises:
            ValueError: If any chunk lacks an embedding
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending

        Raises:
            ValueError: If vector search is not supported
        """
        pass

    def search_fulltext(
        self,
        query_text: str,
        top_k: int = 5
    ) -> list[SearchResult]:
        """Search for chunks using full-text keyword matching.

        Default implementation raises NotImplementedError. Override this
        method if your store supports full-text search (BM25, etc).

        Args:
            query_text: Query text for keyword matching
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending

        Raises:
            NotImplementedError: If full-text search is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support full-text search"
        )

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> list[SearchResult]:
        """Search using hybrid mode (vector + text).

        Default implementation raises NotImplementedError. Override this
        method if your store has native hybrid search support.

        Args:
            query_vector: Query embedding vector
            query_text: Query text for keyword matching
            top_k: Number of results to return
            alpha: Weight for vector vs text scores (0.0=text only, 1.0=vector only)

        Returns:
            List of search results, sorted by combined score descending

        Raises:
            NotImplementedError: If hybrid search is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native hybrid search"
        )

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the total number of chunks in this store.

        Returns:
            Number of chunks currently stored

        Examples:
            >>> store = InMemoryVectorStore()
            >>> store.add(chunks)
            >>> print(f"Total chunks: {store.count()}")
            Total chunks: 100
        """
        pass

    @abstractmethod
    def persist(self, path: str) -> None:
        """Save store to disk.

        Args:
            path: File path to save to
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load store from disk.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        pass
