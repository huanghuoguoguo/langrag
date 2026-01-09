"""
LangRAG Cache Module.

This module provides caching capabilities to reduce redundant
embedding and search operations.

Classes:
    BaseCache: Abstract base class for cache implementations
    CacheEntry: Data class for cache entries
    SemanticCache: Cache that uses embedding similarity for matching

Example:
    >>> from langrag.cache import SemanticCache
    >>> cache = SemanticCache(similarity_threshold=0.95, ttl_seconds=3600)
    >>> cache.set_with_embedding(
    ...     query="What is AI?",
    ...     embedding=[0.1, 0.2, ...],
    ...     results=results,
    ...     metadata={"search_type": "hybrid"}
    ... )
    >>> # Later, with a similar query...
    >>> hit = cache.get_by_similarity(new_embedding)
    >>> if hit:
    ...     return hit.results  # Avoid redundant search
"""

from langrag.cache.base import BaseCache, CacheEntry
from langrag.cache.semantic import SemanticCache, cosine_similarity

__all__ = [
    "BaseCache",
    "CacheEntry",
    "SemanticCache",
    "cosine_similarity",
]
