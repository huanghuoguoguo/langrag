"""Base cache interface for LangRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """
    A single cache entry containing query and results.

    Attributes:
        query: The original query string
        embedding: The query embedding vector
        results: The cached search results
        metadata: Additional metadata (e.g., search_type, kb_id)
        created_at: Unix timestamp of when the entry was created
    """
    query: str
    embedding: list[float]
    results: list[Any]
    metadata: dict[str, Any]
    created_at: float


class BaseCache(ABC):
    """
    Abstract base class for caching implementations.

    Cache implementations store and retrieve query results to avoid
    redundant embedding and search operations.
    """

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """
        Get a cache entry by exact key match.

        Args:
            key: The cache key (typically the query string)

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """
        Store a cache entry.

        Args:
            key: The cache key
            entry: The cache entry to store
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete a cache entry.

        Args:
            key: The cache key to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of entries in the cache."""
        pass
