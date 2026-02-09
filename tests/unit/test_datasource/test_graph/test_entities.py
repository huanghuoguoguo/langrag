"""Unit tests for graph entities."""
import pytest

from langrag.entities.graph import Entity, Relationship, Subgraph


class TestEntity:
    """Tests for Entity model."""

    def test_create_entity_minimal(self):
        """Test creating entity with minimal fields."""
        entity = Entity(name="Test Entity")
        assert entity.name == "Test Entity"
        assert entity.type == "UNKNOWN"
        assert entity.properties == {}
        assert entity.embedding is None
        assert entity.id is not None

    def test_create_entity_full(self):
        """Test creating entity with all fields."""
        entity = Entity(
            id="entity-1",
            name="John Doe",
            type="Person",
            properties={"age": 30, "occupation": "Engineer"},
            embedding=[0.1, 0.2, 0.3],
            source_chunk_ids=["chunk-1", "chunk-2"],
        )
        assert entity.id == "entity-1"
        assert entity.name == "John Doe"
        assert entity.type == "Person"
        assert entity.properties["age"] == 30
        assert entity.embedding == [0.1, 0.2, 0.3]
        assert len(entity.source_chunk_ids) == 2

    def test_get_set_property(self):
        """Test property getter and setter."""
        entity = Entity(name="Test")
        assert entity.get_property("foo") is None
        assert entity.get_property("foo", "default") == "default"

        entity.set_property("foo", "bar")
        assert entity.get_property("foo") == "bar"

    def test_to_text_simple(self):
        """Test text representation without properties."""
        entity = Entity(name="Apple", type="Organization")
        assert entity.to_text() == "Apple (Organization)"

    def test_to_text_with_properties(self):
        """Test text representation with properties."""
        entity = Entity(
            name="John",
            type="Person",
            properties={"role": "CEO"}
        )
        text = entity.to_text()
        assert "John" in text
        assert "Person" in text
        assert "role: CEO" in text


class TestRelationship:
    """Tests for Relationship model."""

    def test_create_relationship_minimal(self):
        """Test creating relationship with minimal fields."""
        rel = Relationship(
            source_id="entity-1",
            target_id="entity-2",
            type="RELATED_TO"
        )
        assert rel.source_id == "entity-1"
        assert rel.target_id == "entity-2"
        assert rel.type == "RELATED_TO"
        assert rel.weight == 1.0
        assert rel.id is not None

    def test_create_relationship_full(self):
        """Test creating relationship with all fields."""
        rel = Relationship(
            id="rel-1",
            source_id="entity-1",
            target_id="entity-2",
            type="WORKS_AT",
            properties={"since": 2020},
            weight=0.9,
            source_chunk_ids=["chunk-1"],
        )
        assert rel.id == "rel-1"
        assert rel.properties["since"] == 2020
        assert rel.weight == 0.9

    def test_to_text(self):
        """Test text representation."""
        rel = Relationship(
            source_id="e1",
            target_id="e2",
            type="KNOWS"
        )
        text = rel.to_text("Alice", "Bob")
        assert text == "Alice --[KNOWS]--> Bob"

    def test_to_text_without_names(self):
        """Test text representation without names uses IDs."""
        rel = Relationship(
            source_id="e1",
            target_id="e2",
            type="KNOWS"
        )
        text = rel.to_text()
        assert text == "e1 --[KNOWS]--> e2"


class TestSubgraph:
    """Tests for Subgraph model."""

    def test_empty_subgraph(self):
        """Test empty subgraph."""
        sg = Subgraph()
        assert sg.is_empty()
        assert sg.to_context() == ""

    def test_subgraph_with_entities(self):
        """Test subgraph with entities only."""
        entities = [
            Entity(id="e1", name="Alice", type="Person"),
            Entity(id="e2", name="Bob", type="Person"),
        ]
        sg = Subgraph(entities=entities)
        assert not sg.is_empty()

        context = sg.to_context()
        assert "## Entities" in context
        assert "Alice" in context
        assert "Bob" in context

    def test_subgraph_with_relationships(self):
        """Test subgraph with entities and relationships."""
        entities = [
            Entity(id="e1", name="Alice", type="Person"),
            Entity(id="e2", name="Bob", type="Person"),
        ]
        relationships = [
            Relationship(id="r1", source_id="e1", target_id="e2", type="KNOWS"),
        ]
        sg = Subgraph(entities=entities, relationships=relationships)

        context = sg.to_context()
        assert "## Entities" in context
        assert "## Relationships" in context
        assert "Alice --[KNOWS]--> Bob" in context

    def test_subgraph_to_context_with_limits(self):
        """Test context generation with limits."""
        entities = [Entity(id=f"e{i}", name=f"Entity{i}", type="Test") for i in range(10)]
        sg = Subgraph(entities=entities)

        context = sg.to_context(max_entities=3)
        assert context.count("Entity") == 3
