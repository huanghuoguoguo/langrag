"""Base vector store interface."""

from abc import ABC, abstractmethod
from ..core.chunk import Chunk
from ..core.search_result import SearchResult


class BaseVectorStore(ABC):
    """Abstract base class for vector storage and similarity search.

    Vector stores persist chunks with their embeddings and provide
    efficient similarity-based retrieval.
    """

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
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
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
