"""Reranker factory for creating reranker instances."""

from typing import Any

from loguru import logger

from .base import BaseReranker
from .providers.cohere import CohereReranker
from .providers.noop import NoOpReranker
from .providers.qwen import QwenReranker


class RerankerFactory:
    """Factory for creating reranker instances based on type.

    This factory maintains a registry of available reranker types
    and creates instances based on string identifiers.
    """

    _registry: dict[str, type[BaseReranker]] = {
        "noop": NoOpReranker,
        "qwen": QwenReranker,
        "cohere": CohereReranker,
    }

    @classmethod
    def create(cls, reranker_type: str, **params: Any) -> BaseReranker:
        """Create a reranker instance by type.

        Args:
            reranker_type: Type identifier (e.g., "noop")
            **params: Initialization parameters for the reranker

        Returns:
            Reranker instance

        Raises:
            ValueError: If reranker type is not registered
        """
        if reranker_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown reranker type: '{reranker_type}'. Available types: {available}"
            )

        reranker_class = cls._registry[reranker_type]
        logger.debug(f"Creating {reranker_class.__name__} with params: {params}")

        return reranker_class(**params)

    @classmethod
    def register(cls, reranker_type: str, reranker_class: type[BaseReranker]):
        """Register a new reranker type.

        Args:
            reranker_type: Type identifier
            reranker_class: Reranker class to register

        Raises:
            TypeError: If reranker_class is not a subclass of BaseReranker
        """
        if not issubclass(reranker_class, BaseReranker):
            raise TypeError(f"{reranker_class.__name__} must be a subclass of BaseReranker")

        cls._registry[reranker_type] = reranker_class
        logger.info(f"Registered reranker type '{reranker_type}': {reranker_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available reranker types.

        Returns:
            List of registered reranker type identifiers
        """
        return list(cls._registry.keys())
