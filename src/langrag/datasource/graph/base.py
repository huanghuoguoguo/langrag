"""Abstract base class for graph stores."""
from abc import ABC, abstractmethod
from typing import Any

from langrag.entities.graph import Entity, Relationship, Subgraph


class BaseGraphStore(ABC):
    """
    Abstract base class for graph storage backends.

    This class defines the interface that all graph store implementations must follow.
    Following LangRAG's "Storage Only" principle, this layer only handles storing and
    retrieving graph data. Entity extraction and other processing logic belongs in
    the IndexProcessor layer.

    Implementations:
        - NetworkXGraphStore: In-memory graph using NetworkX (for development/testing)
        - Neo4jGraphStore: Production graph database (planned)
    """

    @abstractmethod
    async def add_entities(self, entities: list[Entity]) -> None:
        """
        Add entities to the graph store.

        If an entity with the same ID exists, it should be updated (upsert semantics).

        Args:
            entities: List of entities to add
        """
        pass

    @abstractmethod
    async def add_relationships(self, relationships: list[Relationship]) -> None:
        """
        Add relationships to the graph store.

        If a relationship with the same ID exists, it should be updated (upsert semantics).
        Source and target entities must exist before adding relationships.

        Args:
            relationships: List of relationships to add
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Entity | None:
        """
        Get a single entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_entities(self, entity_ids: list[str]) -> list[Entity]:
        """
        Get multiple entities by IDs.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of found entities (missing IDs are silently skipped)
        """
        pass

    @abstractmethod
    async def get_relationship(self, relationship_id: str) -> Relationship | None:
        """
        Get a single relationship by ID.

        Args:
            relationship_id: The relationship ID

        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_ids: list[str],
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both",
    ) -> Subgraph:
        """
        Get neighboring entities via BFS traversal.

        Args:
            entity_ids: Starting entity IDs
            depth: Maximum traversal depth (1 = direct neighbors only)
            relationship_types: Filter by relationship types (None = all types)
            direction: Traversal direction ("in", "out", "both")

        Returns:
            Subgraph containing reached entities and traversed relationships
        """
        pass

    @abstractmethod
    async def search_entities(
        self,
        query: str | None = None,
        query_vector: list[float] | None = None,
        top_k: int = 10,
        entity_types: list[str] | None = None,
        threshold: float = 0.0,
    ) -> list[Entity]:
        """
        Search for entities by text or vector similarity.

        At least one of query or query_vector must be provided.

        Args:
            query: Text query for keyword matching
            query_vector: Vector for similarity search
            top_k: Maximum number of results
            entity_types: Filter by entity types
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of matching entities, sorted by relevance
        """
        pass

    @abstractmethod
    async def delete_entities(self, entity_ids: list[str]) -> int:
        """
        Delete entities and their associated relationships.

        Args:
            entity_ids: IDs of entities to delete

        Returns:
            Number of entities deleted
        """
        pass

    @abstractmethod
    async def delete_relationships(self, relationship_ids: list[str]) -> int:
        """
        Delete relationships by IDs.

        Args:
            relationship_ids: IDs of relationships to delete

        Returns:
            Number of relationships deleted
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entities and relationships from the store."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary with keys like:
            - entity_count: Total number of entities
            - relationship_count: Total number of relationships
            - entity_types: Count by entity type
            - relationship_types: Count by relationship type
        """
        pass

    # Optional: Native query support for advanced use cases
    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        """
        Execute a native query (e.g., Cypher for Neo4j).

        This is optional and may not be supported by all implementations.

        Args:
            query: Native query string
            params: Query parameters

        Returns:
            Query results as list of dictionaries

        Raises:
            NotImplementedError: If native queries are not supported
        """
        raise NotImplementedError("Native queries not supported by this graph store")

    # Sync wrappers for backward compatibility
    def add_entities_sync(self, entities: list[Entity]) -> None:
        """Sync wrapper for add_entities."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.add_entities(entities))

    def add_relationships_sync(self, relationships: list[Relationship]) -> None:
        """Sync wrapper for add_relationships."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.add_relationships(relationships))

    def get_neighbors_sync(
        self,
        entity_ids: list[str],
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both",
    ) -> Subgraph:
        """Sync wrapper for get_neighbors."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.get_neighbors(entity_ids, depth, relationship_types, direction)
        )

    def search_entities_sync(
        self,
        query: str | None = None,
        query_vector: list[float] | None = None,
        top_k: int = 10,
        entity_types: list[str] | None = None,
        threshold: float = 0.0,
    ) -> list[Entity]:
        """Sync wrapper for search_entities."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.search_entities(query, query_vector, top_k, entity_types, threshold)
        )
