"""Vector store module for storage and retrieval.

This module provides vector storage functionality with multiple
implementations and a factory for creating vector stores.
"""

from .base import BaseVectorStore
from .capabilities import VectorStoreCapabilities, SearchMode
from .providers.in_memory import InMemoryVectorStore
from .factory import VectorStoreFactory
from .manager import VectorStoreManager

__all__ = [
    "BaseVectorStore",
    "VectorStoreCapabilities",
    "SearchMode",
    "InMemoryVectorStore",
    "VectorStoreFactory",
    "VectorStoreManager"
]
