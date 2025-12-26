"""Vector store factory for creating vector store instances."""

from typing import Any
from loguru import logger

from .base import BaseVectorStore
from .providers.in_memory import InMemoryVectorStore

# Conditionally import SeekDB if available
try:
    from .providers.seekdb import SeekDBVectorStore
    SEEKDB_AVAILABLE = True
except ImportError:
    SEEKDB_AVAILABLE = False
    logger.debug("SeekDB not available (pyseekdb not installed)")

# Conditionally import Chroma if available
try:
    from .providers.chroma import ChromaVectorStore, CHROMA_AVAILABLE
    CHROMA_VECTORSTORE_AVAILABLE = CHROMA_AVAILABLE
except ImportError:
    CHROMA_VECTORSTORE_AVAILABLE = False
    logger.debug("Chroma not available (chromadb not installed)")


class VectorStoreFactory:
    """Factory for creating vector store instances based on type.

    This factory maintains a registry of available vector store types
    and creates instances based on string identifiers.
    """

    _registry: dict[str, type[BaseVectorStore]] = {
        "in_memory": InMemoryVectorStore,
    }

    # Register SeekDB if available
    if SEEKDB_AVAILABLE:
        _registry["seekdb"] = SeekDBVectorStore

    # Register Chroma if available
    if CHROMA_VECTORSTORE_AVAILABLE:
        _registry["chroma"] = ChromaVectorStore

    @classmethod
    def create(cls, store_type: str, **params: Any) -> BaseVectorStore:
        """Create a vector store instance by type.

        Args:
            store_type: Type identifier (e.g., "in_memory")
            **params: Initialization parameters for the vector store

        Returns:
            Vector store instance

        Raises:
            ValueError: If vector store type is not registered
        """
        if store_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown vector store type: '{store_type}'. "
                f"Available types: {available}"
            )

        store_class = cls._registry[store_type]
        logger.debug(f"Creating {store_class.__name__} with params: {params}")

        return store_class(**params)

    @classmethod
    def register(cls, store_type: str, store_class: type[BaseVectorStore]):
        """Register a new vector store type.

        Args:
            store_type: Type identifier
            store_class: Vector store class to register

        Raises:
            TypeError: If store_class is not a subclass of BaseVectorStore
        """
        if not issubclass(store_class, BaseVectorStore):
            raise TypeError(
                f"{store_class.__name__} must be a subclass of BaseVectorStore"
            )

        cls._registry[store_type] = store_class
        logger.info(f"Registered vector store type '{store_type}': {store_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available vector store types.

        Returns:
            List of registered vector store type identifiers
        """
        return list(cls._registry.keys())
