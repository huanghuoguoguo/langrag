"""Unit tests for NetworkX graph store."""
import pytest

from langrag.entities.graph import Entity, Relationship
from langrag.datasource.graph.networkx import NetworkXGraphStore


@pytest.fixture
def graph_store():
    """Create a fresh graph store for each test."""
    return NetworkXGraphStore()


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(id="e1", name="Alice", type="Person", properties={"age": 30}),
        Entity(id="e2", name="Bob", type="Person", properties={"age": 25}),
        Entity(id="e3", name="Acme Corp", type="Organization"),
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships for testing."""
    return [
        Relationship(id="r1", source_id="e1", target_id="e2", type="KNOWS"),
        Relationship(id="r2", source_id="e1", target_id="e3", type="WORKS_AT"),
        Relationship(id="r3", source_id="e2", target_id="e3", type="WORKS_AT"),
    ]


class TestNetworkXGraphStoreBasics:
    """Basic CRUD tests for NetworkX graph store."""

    @pytest.mark.asyncio
    async def test_add_and_get_entity(self, graph_store, sample_entities):
        """Test adding and retrieving entities."""
        await graph_store.add_entities(sample_entities)

        entity = await graph_store.get_entity("e1")
        assert entity is not None
        assert entity.name == "Alice"
        assert entity.type == "Person"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, graph_store):
        """Test getting a non-existent entity returns None."""
        entity = await graph_store.get_entity("nonexistent")
        assert entity is None

    @pytest.mark.asyncio
    async def test_get_multiple_entities(self, graph_store, sample_entities):
        """Test getting multiple entities."""
        await graph_store.add_entities(sample_entities)

        entities = await graph_store.get_entities(["e1", "e2", "nonexistent"])
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert names == {"Alice", "Bob"}

    @pytest.mark.asyncio
    async def test_add_and_get_relationship(self, graph_store, sample_entities, sample_relationships):
        """Test adding and retrieving relationships."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        rel = await graph_store.get_relationship("r1")
        assert rel is not None
        assert rel.type == "KNOWS"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"

    @pytest.mark.asyncio
    async def test_add_relationship_missing_entity(self, graph_store):
        """Test adding relationship with missing entities raises error."""
        rel = Relationship(id="r1", source_id="e1", target_id="e2", type="KNOWS")

        with pytest.raises(ValueError, match="does not exist"):
            await graph_store.add_relationships([rel])

    @pytest.mark.asyncio
    async def test_upsert_entity(self, graph_store):
        """Test that adding entity with same ID updates it."""
        entity1 = Entity(id="e1", name="Alice", type="Person")
        entity2 = Entity(id="e1", name="Alice Updated", type="Person")

        await graph_store.add_entities([entity1])
        await graph_store.add_entities([entity2])

        entity = await graph_store.get_entity("e1")
        assert entity.name == "Alice Updated"


class TestNetworkXGraphStoreTraversal:
    """Graph traversal tests."""

    @pytest.mark.asyncio
    async def test_get_neighbors_depth_1(self, graph_store, sample_entities, sample_relationships):
        """Test getting direct neighbors."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        subgraph = await graph_store.get_neighbors(["e1"], depth=1)

        entity_ids = {e.id for e in subgraph.entities}
        assert "e1" in entity_ids
        assert "e2" in entity_ids
        assert "e3" in entity_ids

    @pytest.mark.asyncio
    async def test_get_neighbors_direction_out(self, graph_store, sample_entities, sample_relationships):
        """Test outgoing-only traversal."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        subgraph = await graph_store.get_neighbors(["e3"], depth=1, direction="out")

        # e3 has no outgoing edges
        entity_ids = {e.id for e in subgraph.entities}
        assert entity_ids == {"e3"}

    @pytest.mark.asyncio
    async def test_get_neighbors_direction_in(self, graph_store, sample_entities, sample_relationships):
        """Test incoming-only traversal."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        subgraph = await graph_store.get_neighbors(["e3"], depth=1, direction="in")

        entity_ids = {e.id for e in subgraph.entities}
        assert "e1" in entity_ids
        assert "e2" in entity_ids
        assert "e3" in entity_ids

    @pytest.mark.asyncio
    async def test_get_neighbors_filter_by_type(self, graph_store, sample_entities, sample_relationships):
        """Test filtering by relationship type."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        subgraph = await graph_store.get_neighbors(
            ["e1"], depth=1, relationship_types=["KNOWS"]
        )

        entity_ids = {e.id for e in subgraph.entities}
        assert "e2" in entity_ids
        assert "e3" not in entity_ids  # WORKS_AT filtered out


class TestNetworkXGraphStoreSearch:
    """Search tests."""

    @pytest.mark.asyncio
    async def test_search_by_text(self, graph_store, sample_entities):
        """Test text-based search."""
        await graph_store.add_entities(sample_entities)

        results = await graph_store.search_entities(query="Alice")
        assert len(results) == 1
        assert results[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_search_by_text_case_insensitive(self, graph_store, sample_entities):
        """Test case-insensitive search."""
        await graph_store.add_entities(sample_entities)

        results = await graph_store.search_entities(query="alice")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_by_vector(self, graph_store):
        """Test vector-based search."""
        entities = [
            Entity(id="e1", name="A", type="Test", embedding=[1.0, 0.0, 0.0]),
            Entity(id="e2", name="B", type="Test", embedding=[0.0, 1.0, 0.0]),
            Entity(id="e3", name="C", type="Test", embedding=[0.9, 0.1, 0.0]),
        ]
        await graph_store.add_entities(entities)

        # Search for vector similar to e1 and e3
        results = await graph_store.search_entities(
            query_vector=[1.0, 0.0, 0.0], top_k=2
        )

        names = [r.name for r in results]
        assert "A" in names
        assert "C" in names

    @pytest.mark.asyncio
    async def test_search_filter_by_type(self, graph_store, sample_entities):
        """Test search with type filter."""
        await graph_store.add_entities(sample_entities)

        results = await graph_store.search_entities(
            query="A", entity_types=["Organization"]
        )
        assert len(results) == 1
        assert results[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_search_requires_query_or_vector(self, graph_store):
        """Test that search requires query or vector."""
        with pytest.raises(ValueError, match="At least one"):
            await graph_store.search_entities()


class TestNetworkXGraphStoreDelete:
    """Delete operation tests."""

    @pytest.mark.asyncio
    async def test_delete_entity(self, graph_store, sample_entities, sample_relationships):
        """Test deleting an entity removes it and its relationships."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        deleted = await graph_store.delete_entities(["e1"])
        assert deleted == 1

        # Entity should be gone
        assert await graph_store.get_entity("e1") is None

        # Relationships involving e1 should be gone
        assert await graph_store.get_relationship("r1") is None
        assert await graph_store.get_relationship("r2") is None

        # Other relationships should remain
        assert await graph_store.get_relationship("r3") is not None

    @pytest.mark.asyncio
    async def test_delete_relationship(self, graph_store, sample_entities, sample_relationships):
        """Test deleting a relationship."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        deleted = await graph_store.delete_relationships(["r1"])
        assert deleted == 1

        # Relationship should be gone
        assert await graph_store.get_relationship("r1") is None

        # Entities should remain
        assert await graph_store.get_entity("e1") is not None
        assert await graph_store.get_entity("e2") is not None

    @pytest.mark.asyncio
    async def test_clear(self, graph_store, sample_entities, sample_relationships):
        """Test clearing the graph."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        await graph_store.clear()

        stats = await graph_store.get_stats()
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0


class TestNetworkXGraphStoreStats:
    """Statistics tests."""

    @pytest.mark.asyncio
    async def test_get_stats(self, graph_store, sample_entities, sample_relationships):
        """Test getting graph statistics."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        stats = await graph_store.get_stats()

        assert stats["entity_count"] == 3
        assert stats["relationship_count"] == 3
        assert stats["entity_types"]["Person"] == 2
        assert stats["entity_types"]["Organization"] == 1
        assert stats["relationship_types"]["KNOWS"] == 1
        assert stats["relationship_types"]["WORKS_AT"] == 2


class TestNetworkXGraphStoreUtilities:
    """Utility method tests."""

    @pytest.mark.asyncio
    async def test_get_connected_components(self, graph_store):
        """Test finding connected components."""
        # Create two disconnected components
        entities = [
            Entity(id="e1", name="A", type="Test"),
            Entity(id="e2", name="B", type="Test"),
            Entity(id="e3", name="C", type="Test"),
            Entity(id="e4", name="D", type="Test"),
        ]
        relationships = [
            Relationship(id="r1", source_id="e1", target_id="e2", type="CONNECTED"),
            Relationship(id="r2", source_id="e3", target_id="e4", type="CONNECTED"),
        ]
        await graph_store.add_entities(entities)
        await graph_store.add_relationships(relationships)

        components = await graph_store.get_connected_components()
        assert len(components) == 2

    @pytest.mark.asyncio
    async def test_get_shortest_path(self, graph_store, sample_entities, sample_relationships):
        """Test finding shortest path."""
        await graph_store.add_entities(sample_entities)
        await graph_store.add_relationships(sample_relationships)

        path = await graph_store.get_shortest_path("e1", "e2")
        assert path is not None
        assert path == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_get_shortest_path_no_path(self, graph_store):
        """Test shortest path when no path exists."""
        entities = [
            Entity(id="e1", name="A", type="Test"),
            Entity(id="e2", name="B", type="Test"),
        ]
        await graph_store.add_entities(entities)

        path = await graph_store.get_shortest_path("e1", "e2")
        assert path is None
