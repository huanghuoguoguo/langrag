"""Unit tests for graph store factory."""
import pytest

from langrag.datasource.graph.factory import GraphStoreFactory
from langrag.datasource.graph.networkx import NetworkXGraphStore


class TestGraphStoreFactory:
    """Tests for GraphStoreFactory."""

    def test_create_networkx(self):
        """Test creating NetworkX graph store."""
        store = GraphStoreFactory.create("networkx")
        assert isinstance(store, NetworkXGraphStore)

    def test_create_networkx_case_insensitive(self):
        """Test that store type is case-insensitive."""
        store = GraphStoreFactory.create("NetworkX")
        assert isinstance(store, NetworkXGraphStore)

    def test_create_neo4j_not_implemented(self):
        """Test that Neo4j raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Neo4j"):
            GraphStoreFactory.create("neo4j")

    def test_create_unknown_type(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown graph store type"):
            GraphStoreFactory.create("unknown")

    def test_available_types(self):
        """Test listing available types."""
        types = GraphStoreFactory.available_types()
        assert "networkx" in types
        assert "neo4j" in types

    def test_register_custom_store(self):
        """Test registering a custom store."""
        # Register a custom store
        GraphStoreFactory.register("custom", NetworkXGraphStore)

        store = GraphStoreFactory.create("custom")
        assert isinstance(store, NetworkXGraphStore)

        # Cleanup
        del GraphStoreFactory._registry["custom"]
