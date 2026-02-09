"""NetworkX-based in-memory graph store implementation."""
from collections import defaultdict
from typing import Any

import networkx as nx

from langrag.entities.graph import Entity, Relationship, Subgraph
from langrag.datasource.graph.base import BaseGraphStore


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class NetworkXGraphStore(BaseGraphStore):
    """
    In-memory graph store using NetworkX.

    This implementation is suitable for development, testing, and small-scale
    production use cases. For large-scale deployments, use Neo4jGraphStore.

    Features:
        - Full graph traversal support
        - Vector similarity search on entity embeddings
        - Text-based entity search
        - Thread-safe for read operations

    Limitations:
        - Data is not persisted (lost on restart)
        - Not suitable for very large graphs (>1M nodes)
        - Single-process only
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}

    async def add_entities(self, entities: list[Entity]) -> None:
        """Add entities to the graph (upsert semantics)."""
        for entity in entities:
            self._entities[entity.id] = entity
            self._graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                properties=entity.properties,
                embedding=entity.embedding,
                source_chunk_ids=entity.source_chunk_ids,
            )

    async def add_relationships(self, relationships: list[Relationship]) -> None:
        """Add relationships to the graph (upsert semantics)."""
        for rel in relationships:
            # Ensure source and target nodes exist
            if rel.source_id not in self._graph:
                raise ValueError(f"Source entity {rel.source_id} does not exist")
            if rel.target_id not in self._graph:
                raise ValueError(f"Target entity {rel.target_id} does not exist")

            self._relationships[rel.id] = rel
            self._graph.add_edge(
                rel.source_id,
                rel.target_id,
                id=rel.id,
                type=rel.type,
                properties=rel.properties,
                weight=rel.weight,
                source_chunk_ids=rel.source_chunk_ids,
            )

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get a single entity by ID."""
        return self._entities.get(entity_id)

    async def get_entities(self, entity_ids: list[str]) -> list[Entity]:
        """Get multiple entities by IDs."""
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]

    async def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Get a single relationship by ID."""
        return self._relationships.get(relationship_id)

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
            depth: Maximum traversal depth
            relationship_types: Filter by relationship types
            direction: "in", "out", or "both"

        Returns:
            Subgraph with entities and relationships
        """
        if depth < 1:
            depth = 1

        visited_entities: set[str] = set()
        visited_relationships: set[str] = set()
        current_level = set(eid for eid in entity_ids if eid in self._graph)

        # Add starting nodes to visited
        visited_entities.update(current_level)

        for _ in range(depth):
            next_level: set[str] = set()

            for node_id in current_level:
                # Outgoing edges
                if direction in ("out", "both"):
                    for _, target, edge_data in self._graph.out_edges(node_id, data=True):
                        rel_type = edge_data.get("type", "")
                        if relationship_types is None or rel_type in relationship_types:
                            rel_id = edge_data.get("id")
                            if rel_id:
                                visited_relationships.add(rel_id)
                            if target not in visited_entities:
                                next_level.add(target)
                                visited_entities.add(target)

                # Incoming edges
                if direction in ("in", "both"):
                    for source, _, edge_data in self._graph.in_edges(node_id, data=True):
                        rel_type = edge_data.get("type", "")
                        if relationship_types is None or rel_type in relationship_types:
                            rel_id = edge_data.get("id")
                            if rel_id:
                                visited_relationships.add(rel_id)
                            if source not in visited_entities:
                                next_level.add(source)
                                visited_entities.add(source)

            current_level = next_level

        # Build result
        entities = [self._entities[eid] for eid in visited_entities if eid in self._entities]
        relationships = [self._relationships[rid] for rid in visited_relationships if rid in self._relationships]

        return Subgraph(entities=entities, relationships=relationships)

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

        Args:
            query: Text query for keyword matching (case-insensitive)
            query_vector: Vector for similarity search
            top_k: Maximum number of results
            entity_types: Filter by entity types
            threshold: Minimum similarity score

        Returns:
            List of matching entities, sorted by relevance
        """
        if query is None and query_vector is None:
            raise ValueError("At least one of query or query_vector must be provided")

        candidates: list[tuple[float, Entity]] = []

        for entity in self._entities.values():
            # Filter by type
            if entity_types and entity.type not in entity_types:
                continue

            score = 0.0

            # Text matching
            if query:
                query_lower = query.lower()
                name_lower = entity.name.lower()

                if query_lower in name_lower:
                    # Exact substring match gets high score
                    score = max(score, 0.8 + 0.2 * (len(query_lower) / len(name_lower)))
                elif any(query_lower in str(v).lower() for v in entity.properties.values()):
                    # Property match gets medium score
                    score = max(score, 0.5)

            # Vector similarity
            if query_vector and entity.embedding:
                vec_score = _cosine_similarity(query_vector, entity.embedding)
                score = max(score, vec_score)

            if score > threshold or (score > 0 and threshold == 0.0):
                candidates.append((score, entity))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [entity for _, entity in candidates[:top_k]]

    async def delete_entities(self, entity_ids: list[str]) -> int:
        """Delete entities and their associated relationships."""
        deleted = 0
        for entity_id in entity_ids:
            if entity_id in self._entities:
                # Remove associated relationships
                rels_to_delete = [
                    rid for rid, rel in self._relationships.items()
                    if rel.source_id == entity_id or rel.target_id == entity_id
                ]
                for rid in rels_to_delete:
                    del self._relationships[rid]

                # Remove entity
                del self._entities[entity_id]
                if entity_id in self._graph:
                    self._graph.remove_node(entity_id)
                deleted += 1

        return deleted

    async def delete_relationships(self, relationship_ids: list[str]) -> int:
        """Delete relationships by IDs."""
        deleted = 0
        for rel_id in relationship_ids:
            if rel_id in self._relationships:
                rel = self._relationships[rel_id]
                del self._relationships[rel_id]

                # Remove edge from graph
                if self._graph.has_edge(rel.source_id, rel.target_id):
                    self._graph.remove_edge(rel.source_id, rel.target_id)
                deleted += 1

        return deleted

    async def clear(self) -> None:
        """Clear all entities and relationships."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        entity_types: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            entity_types[entity.type] += 1

        relationship_types: dict[str, int] = defaultdict(int)
        for rel in self._relationships.values():
            relationship_types[rel.type] += 1

        return {
            "entity_count": len(self._entities),
            "relationship_count": len(self._relationships),
            "entity_types": dict(entity_types),
            "relationship_types": dict(relationship_types),
        }

    # Additional utility methods for NetworkX

    def get_networkx_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph for advanced operations."""
        return self._graph

    async def get_connected_components(self) -> list[set[str]]:
        """Get weakly connected components."""
        undirected = self._graph.to_undirected()
        return [set(component) for component in nx.connected_components(undirected)]

    async def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """
        Get shortest path between two entities.

        Returns:
            List of entity IDs forming the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self._graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
