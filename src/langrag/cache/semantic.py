"""
Semantic Cache Implementation.

This module provides a cache that uses embedding similarity to match queries,
allowing semantically similar queries to return cached results.

Features:
- Cosine similarity matching for semantic query comparison
- Configurable similarity threshold
- TTL (time-to-live) support for cache entries
- LRU eviction when max size is reached
- Thread-safe operations
- NumPy-accelerated similarity computation (with pure Python fallback)

Example:
    >>> from langrag.cache import SemanticCache
    >>> cache = SemanticCache(similarity_threshold=0.95, ttl_seconds=3600)
    >>> cache.set_with_embedding(query, embedding, results, metadata)
    >>> hit = cache.get_by_similarity(new_embedding)
    >>> if hit:
    ...     return hit.results
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from langrag.cache.base import BaseCache, CacheEntry

logger = logging.getLogger(__name__)

# Try to import NumPy for accelerated similarity computation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.debug("NumPy available: using accelerated cosine similarity")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available: using pure Python cosine similarity")


def _cosine_similarity_numpy(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity using NumPy (fast).

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if len(vec1) != len(vec2):
        return 0.0

    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def _cosine_similarity_python(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity using pure Python (portable fallback).

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Automatically uses NumPy if available for ~100x speedup on large vectors.
    Falls back to pure Python implementation for portability.

    Performance comparison (1000-dim vectors, 1000 comparisons):
    - Pure Python: ~100ms
    - NumPy: ~1ms

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if NUMPY_AVAILABLE:
        return _cosine_similarity_numpy(vec1, vec2)
    return _cosine_similarity_python(vec1, vec2)


class SemanticCache(BaseCache):
    """
    Cache that matches queries based on embedding similarity.

    This cache stores query embeddings alongside results and uses
    cosine similarity to find matching cached entries for new queries.

    Attributes:
        similarity_threshold: Minimum similarity score for a cache hit (0.0-1.0)
        ttl_seconds: Time-to-live in seconds (0 = no expiration)
        max_size: Maximum number of entries (0 = unlimited)

    Example:
        >>> cache = SemanticCache(
        ...     similarity_threshold=0.95,
        ...     ttl_seconds=3600,
        ...     max_size=1000
        ... )
        >>> # Store a result
        >>> cache.set_with_embedding(
        ...     query="What is machine learning?",
        ...     embedding=[0.1, 0.2, ...],
        ...     results=[doc1, doc2],
        ...     metadata={"search_type": "hybrid"}
        ... )
        >>> # Retrieve by similarity
        >>> hit = cache.get_by_similarity([0.1, 0.2, ...])
        >>> if hit:
        ...     print(f"Cache hit! Query: {hit.query}")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 1000
    ):
        """
        Initialize the semantic cache.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hit.
                Higher values (e.g., 0.98) require near-exact matches.
                Lower values (e.g., 0.90) allow more semantic variation.
                Default: 0.95
            ttl_seconds: Cache entry lifetime in seconds.
                0 means no expiration.
                Default: 3600 (1 hour)
            max_size: Maximum number of cached entries.
                When exceeded, oldest entries are evicted (LRU).
                0 means unlimited.
                Default: 1000
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        # OrderedDict for LRU eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Stats
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CacheEntry | None:
        """
        Get a cache entry by exact key match.

        Args:
            key: The exact query string to look up

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU
            self._cache.move_to_end(key)
            self._hits += 1
            return entry

    def get_by_similarity(
        self,
        embedding: list[float],
        context_key: str | None = None
    ) -> CacheEntry | None:
        """
        Find a cached entry by embedding similarity.

        This is the primary method for semantic caching. It searches
        all cached entries to find one with similar embedding.

        Args:
            embedding: Query embedding vector to match against
            context_key: Optional context (e.g., kb_id) to narrow search

        Returns:
            CacheEntry if a similar query is found, None otherwise
        """
        with self._lock:
            best_match: CacheEntry | None = None
            best_score = 0.0
            expired_keys = []

            for key, entry in self._cache.items():
                # Check TTL
                if self._is_expired(entry):
                    expired_keys.append(key)
                    continue

                # Optional context filtering
                if context_key and entry.metadata.get("context_key") != context_key:
                    continue

                # Compute similarity
                score = cosine_similarity(embedding, entry.embedding)

                if score >= self.similarity_threshold and score > best_score:
                    best_score = score
                    best_match = entry

            # Clean up expired entries
            for key in expired_keys:
                del self._cache[key]

            if best_match:
                self._hits += 1
                logger.debug(
                    f"Semantic cache hit: score={best_score:.4f}, "
                    f"query='{best_match.query[:50]}...'"
                )
                # Move to end for LRU
                for key, entry in self._cache.items():
                    if entry is best_match:
                        self._cache.move_to_end(key)
                        break
            else:
                self._misses += 1

            return best_match

    def set(self, key: str, entry: CacheEntry) -> None:
        """
        Store a cache entry with the given key.

        Args:
            key: The cache key
            entry: The cache entry to store
        """
        with self._lock:
            # Evict if at max size
            self._evict_if_needed()

            self._cache[key] = entry
            self._cache.move_to_end(key)

    def set_with_embedding(
        self,
        query: str,
        embedding: list[float],
        results: list[Any],
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Store a query result with its embedding.

        This is the primary method for storing semantic cache entries.

        Args:
            query: The original query string
            embedding: The query embedding vector
            results: The search results to cache
            metadata: Optional metadata (search_type, kb_id, etc.)
        """
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            results=results,
            metadata=metadata or {},
            created_at=time.time()
        )
        self.set(query, entry)
        logger.debug(f"Cached query: '{query[:50]}...'")

    def delete(self, key: str) -> None:
        """Delete a cache entry by key."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, and size
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": self.size,
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - entry.created_at > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is at max size."""
        if self.max_size <= 0:
            return

        while len(self._cache) >= self.max_size:
            # Remove oldest (first) item
            self._cache.popitem(last=False)
            logger.debug("Evicted oldest cache entry (LRU)")
