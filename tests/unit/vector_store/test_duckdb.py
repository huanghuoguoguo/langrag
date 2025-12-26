"""Tests for DuckDB vector store."""

import tempfile
from pathlib import Path

import pytest

from langrag.core.chunk import Chunk
from langrag.core.search_result import SearchResult
from langrag.vector_store.factory import VectorStoreFactory

try:
    from langrag.vector_store.providers.duckdb import DUCKDB_AVAILABLE
except ImportError:
    DUCKDB_AVAILABLE = False


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available (duckdb not installed)")
class TestDuckDBVectorStore:
    """Test DuckDB vector store functionality."""

    def test_duckdb_registration(self):
        """Test that DuckDB is registered in the factory."""
        assert "duckdb" in VectorStoreFactory.list_types()

    def test_duckdb_in_memory_creation(self):
        """Test creating DuckDB store in memory."""
        store = VectorStoreFactory.create("duckdb", database_path=":memory:")
        assert store.capabilities.supports_vector
        assert store.capabilities.supports_fulltext
        assert not store.capabilities.supports_hybrid

    def test_duckdb_persistent_creation(self):
        """Test creating DuckDB store with persistent database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            _ = VectorStoreFactory.create("duckdb", database_path=str(db_path))
            assert db_path.exists()

    def test_duckdb_add_and_search(self):
        """Test adding chunks and searching."""
        # Create store with smaller dimension for testing
        dim = 3
        store = VectorStoreFactory.create("duckdb", database_path=":memory:", vector_dimension=dim)

        # Create test chunks
        chunks = [
            Chunk(
                id="doc1",
                content="Python is a programming language",
                embedding=[1.0, 0.0, 0.0],  # Unit vector in x direction
                source_doc_id="source1",
                metadata={"type": "programming"},
            ),
            Chunk(
                id="doc2",
                content="Machine learning with Python",
                embedding=[0.0, 1.0, 0.0],  # Unit vector in y direction
                source_doc_id="source2",
                metadata={"type": "ml"},
            ),
            Chunk(
                id="doc3",
                content="Data science fundamentals",
                embedding=[0.0, 0.0, 1.0],  # Unit vector in z direction
                source_doc_id="source3",
                metadata={"type": "data_science"},
            ),
        ]

        # Add chunks
        store.add(chunks)

        # Test vector search - query should match doc1 best (cosine similarity = 1.0)
        query_vector = [1.0, 0.0, 0.0]
        results = store.search(query_vector, top_k=2)

        assert len(results) == 2
        assert results[0].chunk.id == "doc1"  # Most similar
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0.0 for r in results)

    def test_duckdb_fulltext_search(self):
        """Test full-text search functionality."""
        dim = 3
        store = VectorStoreFactory.create("duckdb", database_path=":memory:", vector_dimension=dim)

        # Create test chunks
        chunks = [
            Chunk(
                id="doc1",
                content="Python is a high-level programming language",
                embedding=[1.0, 0.0, 0.0],
                source_doc_id="source1",
            ),
            Chunk(
                id="doc2",
                content="Machine learning algorithms are powerful",
                embedding=[0.0, 1.0, 0.0],
                source_doc_id="source2",
            ),
            Chunk(
                id="doc3",
                content="Data structures in computer science",
                embedding=[0.0, 0.0, 1.0],
                source_doc_id="source3",
            ),
        ]

        # Add chunks
        store.add(chunks)

        # Test full-text search
        results = store.search_fulltext("Python programming", top_k=2)

        assert len(results) <= 2  # May return fewer results if no matches
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0.0 for r in results)

        # Test search for "machine learning"
        results = store.search_fulltext("machine learning", top_k=2)
        assert len(results) <= 2

    def test_duckdb_delete(self):
        """Test deleting chunks."""
        dim = 3
        store = VectorStoreFactory.create("duckdb", database_path=":memory:", vector_dimension=dim)

        # Add chunks
        chunks = [
            Chunk(
                id="doc1", content="Test content 1", embedding=[1.0, 0.0, 0.0], source_doc_id="src1"
            ),
            Chunk(
                id="doc2", content="Test content 2", embedding=[0.0, 1.0, 0.0], source_doc_id="src2"
            ),
        ]
        store.add(chunks)

        # Verify chunks exist
        results = store.search([1.0, 0.0, 0.0], top_k=10)
        assert len(results) == 2

        # Delete one chunk
        store.delete(["doc1"])

        # Verify deletion
        results = store.search([1.0, 0.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].chunk.id == "doc2"

    def test_duckdb_persist_and_load(self):
        """Test persisting and loading database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create and populate store
            dim = 3
            store1 = VectorStoreFactory.create(
                "duckdb", database_path=str(db_path), vector_dimension=dim
            )
            chunk = Chunk(
                id="doc1", content="Test content", embedding=[1.0, 0.0, 0.0], source_doc_id="src1"
            )
            store1.add([chunk])

            # Persist (checkpoint)
            store1.persist(str(db_path))

            # Create new store and load
            store2 = VectorStoreFactory.create("duckdb", database_path=":memory:")
            store2.load(str(db_path))

            # Verify data loaded
            results = store2.search([1.0, 0.0, 0.0], top_k=1)
            assert len(results) == 1
            assert results[0].chunk.id == "doc1"

    def test_duckdb_metadata_handling(self):
        """Test metadata storage and retrieval."""
        dim = 3
        store = VectorStoreFactory.create("duckdb", database_path=":memory:", vector_dimension=dim)

        # Add chunk with metadata
        metadata = {"author": "Test Author", "year": 2024, "tags": ["test", "example"]}
        chunk = Chunk(
            id="doc1",
            content="Test content",
            embedding=[1.0, 0.0, 0.0],
            source_doc_id="src1",
            metadata=metadata,
        )
        store.add([chunk])

        # Search and verify metadata
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        retrieved_metadata = results[0].chunk.metadata
        assert retrieved_metadata["author"] == "Test Author"
        assert retrieved_metadata["year"] == 2024
        assert retrieved_metadata["tags"] == ["test", "example"]

    def test_duckdb_hybrid_not_supported(self):
        """Test that hybrid search raises NotImplementedError."""
        store = VectorStoreFactory.create("duckdb", database_path=":memory:")

        with pytest.raises(NotImplementedError):
            store.search_hybrid([1.0, 0.0, 0.0], "test query")

    def test_duckdb_empty_search(self):
        """Test searching empty store."""
        store = VectorStoreFactory.create("duckdb", database_path=":memory:")

        # Vector search on empty store
        results = store.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []

        # Full-text search on empty store
        results = store.search_fulltext("test query", top_k=5)
        assert results == []
