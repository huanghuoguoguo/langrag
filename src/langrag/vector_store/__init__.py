"""Vector store module for storage and retrieval.

This module provides vector storage functionality with multiple
implementations and a factory for creating vector stores.
"""

from .base import BaseVectorStore
from .capabilities import SearchMode, VectorStoreCapabilities
from .factory import VectorStoreFactory
from .manager import VectorStoreManager
from .providers.in_memory import InMemoryVectorStore

__all__ = [
    "BaseVectorStore",
    "VectorStoreCapabilities",
    "SearchMode",
    "InMemoryVectorStore",
    "VectorStoreFactory",
    "VectorStoreManager",
]
