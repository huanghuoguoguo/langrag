"""Chunker factory for creating chunker instances."""

from typing import Any
from loguru import logger

from .base import BaseChunker
from .providers.fixed_size import FixedSizeChunker
from .providers.recursive_character import RecursiveCharacterChunker


class ChunkerFactory:
    """Factory for creating chunker instances based on type.

    This factory maintains a registry of available chunker types
    and creates instances based on string identifiers.
    """

    _registry: dict[str, type[BaseChunker]] = {
        "fixed_size": FixedSizeChunker,
        "recursive": RecursiveCharacterChunker,
        # Future chunkers can be registered here:
        # "semantic": SemanticChunker,
        # "sentence": SentenceChunker,
    }

    @classmethod
    def create(cls, chunker_type: str, **params: Any) -> BaseChunker:
        """Create a chunker instance by type.

        Args:
            chunker_type: Type identifier (e.g., "fixed_size")
            **params: Initialization parameters for the chunker

        Returns:
            Chunker instance

        Raises:
            ValueError: If chunker type is not registered
        """
        if chunker_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown chunker type: '{chunker_type}'. "
                f"Available types: {available}"
            )

        chunker_class = cls._registry[chunker_type]
        logger.debug(f"Creating {chunker_class.__name__} with params: {params}")

        return chunker_class(**params)

    @classmethod
    def register(cls, chunker_type: str, chunker_class: type[BaseChunker]):
        """Register a new chunker type.

        Args:
            chunker_type: Type identifier
            chunker_class: Chunker class to register

        Raises:
            TypeError: If chunker_class is not a subclass of BaseChunker
        """
        if not issubclass(chunker_class, BaseChunker):
            raise TypeError(
                f"{chunker_class.__name__} must be a subclass of BaseChunker"
            )

        cls._registry[chunker_type] = chunker_class
        logger.info(f"Registered chunker type '{chunker_type}': {chunker_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available chunker types.

        Returns:
            List of registered chunker type identifiers
        """
        return list(cls._registry.keys())
