"""Factory for creating graph store instances."""
from typing import Any

from langrag.datasource.graph.base import BaseGraphStore


class GraphStoreFactory:
    """
    Factory for creating graph store instances.

    Supports creating different graph store backends based on configuration.

    Example:
        >>> store = GraphStoreFactory.create("networkx")
        >>> store = GraphStoreFactory.create("neo4j", uri="bolt://localhost:7687")
    """

    _registry: dict[str, type[BaseGraphStore]] = {}

    @classmethod
    def register(cls, name: str, store_class: type[BaseGraphStore]) -> None:
        """
        Register a graph store implementation.

        Args:
            name: Store type name (e.g., "networkx", "neo4j")
            store_class: The graph store class
        """
        cls._registry[name.lower()] = store_class

    @classmethod
    def create(cls, store_type: str, **kwargs: Any) -> BaseGraphStore:
        """
        Create a graph store instance.

        Args:
            store_type: Type of graph store ("networkx", "neo4j")
            **kwargs: Additional arguments passed to the store constructor

        Returns:
            BaseGraphStore instance

        Raises:
            ValueError: If store_type is not supported
        """
        store_type_lower = store_type.lower()

        # Lazy import to avoid circular dependencies and optional deps
        if store_type_lower == "networkx":
            from langrag.datasource.graph.networkx import NetworkXGraphStore
            return NetworkXGraphStore(**kwargs)

        elif store_type_lower == "neo4j":
            # Neo4j implementation (planned)
            raise NotImplementedError(
                "Neo4j graph store is not yet implemented. "
                "Use 'networkx' for development or contribute to issue #69."
            )

        elif store_type_lower in cls._registry:
            return cls._registry[store_type_lower](**kwargs)

        else:
            available = ["networkx", "neo4j"] + list(cls._registry.keys())
            raise ValueError(
                f"Unknown graph store type: {store_type}. "
                f"Available types: {', '.join(available)}"
            )

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of available graph store types."""
        return ["networkx", "neo4j"] + list(cls._registry.keys())
