"""Embedder factory for creating embedder instances."""

from typing import Any
from loguru import logger

from .base import BaseEmbedder
from .providers.mock import MockEmbedder

# Conditionally import SeekDB embedder if available
try:
    from .providers.seekdb_embedder import SeekDBEmbedder, SEEKDB_AVAILABLE
    SEEKDB_EMBEDDER_AVAILABLE = SEEKDB_AVAILABLE
except ImportError:
    SEEKDB_EMBEDDER_AVAILABLE = False
    logger.debug("SeekDB embedder not available (pyseekdb not installed)")


class EmbedderFactory:
    """Factory for creating embedder instances based on type.

    This factory maintains a registry of available embedder types
    and creates instances based on string identifiers.
    """

    _registry: dict[str, type[BaseEmbedder]] = {
        "mock": MockEmbedder,
    }

    # Register SeekDB embedder only if pyseekdb is actually available
    if SEEKDB_EMBEDDER_AVAILABLE:
        _registry["seekdb"] = SeekDBEmbedder

    @classmethod
    def create(cls, embedder_type: str, **params: Any) -> BaseEmbedder:
        """Create an embedder instance by type.

        Args:
            embedder_type: Type identifier (e.g., "mock")
            **params: Initialization parameters for the embedder

        Returns:
            Embedder instance

        Raises:
            ValueError: If embedder type is not registered
        """
        if embedder_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown embedder type: '{embedder_type}'. "
                f"Available types: {available}"
            )

        embedder_class = cls._registry[embedder_type]
        logger.debug(f"Creating {embedder_class.__name__} with params: {params}")

        return embedder_class(**params)

    @classmethod
    def register(cls, embedder_type: str, embedder_class: type[BaseEmbedder]):
        """Register a new embedder type.

        Args:
            embedder_type: Type identifier
            embedder_class: Embedder class to register

        Raises:
            TypeError: If embedder_class is not a subclass of BaseEmbedder
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise TypeError(
                f"{embedder_class.__name__} must be a subclass of BaseEmbedder"
            )

        cls._registry[embedder_type] = embedder_class
        logger.info(f"Registered embedder type '{embedder_type}': {embedder_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available embedder types.

        Returns:
            List of registered embedder type identifiers
        """
        return list(cls._registry.keys())
