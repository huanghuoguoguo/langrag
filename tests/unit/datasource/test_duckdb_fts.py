"""
Tests for DuckDB Vector Store with Full-Text Search support.

These tests verify:
1. Basic vector search functionality
2. Full-Text Search (FTS) with BM25 ranking
3. Hybrid search combining vector + FTS
4. Document CRUD operations
"""

import tempfile
from pathlib import Path

import pytest

from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


@pytest.fixture
def dataset():
    """Create a test dataset."""
    return Dataset(
        id="test-kb",
        tenant_id="test-tenant",
        name="Test Knowledge Base",
        description="Test dataset for DuckDB FTS",
        indexing_technique="high_quality",
        collection_name="test_collection",
        vdb_type="duckdb"
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test.duckdb")


@pytest.fixture
def sample_documents():
    """Create sample documents with vectors."""
    return [
        Document(
            id="doc1",
            page_content="Python is a popular programming language for machine learning.",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "programming.txt"}
        ),
        Document(
            id="doc2",
            page_content="Machine learning algorithms learn patterns from data automatically.",
            vector=[0.2, 0.3, 0.4, 0.5],
            metadata={"source": "ml.txt"}
        ),
        Document(
            id="doc3",
            page_content="JavaScript is used for web development and runs in browsers.",
            vector=[0.5, 0.4, 0.3, 0.2],
            metadata={"source": "web.txt"}
        ),
        Document(
            id="doc4",
            page_content="Deep learning uses neural networks for complex pattern recognition.",
            vector=[0.15, 0.25, 0.35, 0.45],
            metadata={"source": "deep_learning.txt"}
        ),
    ]


class TestDuckDBVector:
    """Tests for DuckDB vector store functionality."""

    def test_create_and_add_texts(self, dataset, temp_db_path, sample_documents):
        """Test creating a collection and adding documents."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Verify documents were added by searching
        results = store._search_vector([0.1, 0.2, 0.3, 0.4], top_k=4)
        assert len(results) == 4

    def test_vector_search(self, dataset, temp_db_path, sample_documents):
        """Test vector similarity search."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Search with a vector similar to doc1
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = store.search("", query_vector, top_k=2)

        assert len(results) == 2
        # doc1 should be the most similar (exact match vector)
        assert results[0].id == "doc1"
        assert "score" in results[0].metadata

    def test_keyword_search_fts(self, dataset, temp_db_path, sample_documents):
        """Test Full-Text Search with BM25 ranking."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Search for "machine learning"
        results = store.search("machine learning", None, top_k=4, search_type="keyword")

        assert len(results) > 0
        # Documents about machine learning should rank higher
        content_texts = [r.page_content.lower() for r in results]
        assert any("machine" in t or "learning" in t for t in content_texts)

    def test_keyword_search_specific_term(self, dataset, temp_db_path, sample_documents):
        """Test FTS for a specific term that appears in only one document."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Search for "javascript" - only in doc3
        results = store.search("javascript", None, top_k=4, search_type="keyword")

        assert len(results) >= 1
        assert results[0].id == "doc3"

    def test_hybrid_search(self, dataset, temp_db_path, sample_documents):
        """Test hybrid search combining vector and FTS."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Hybrid search: vector similar to doc1, query about "machine learning"
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = store.search("machine learning", query_vector, top_k=3, search_type="hybrid")

        assert len(results) > 0
        # Results should have hybrid search metadata
        assert "score" in results[0].metadata

    def test_delete_by_ids(self, dataset, temp_db_path, sample_documents):
        """Test deleting documents by ID."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Delete doc1 and doc2
        store.delete_by_ids(["doc1", "doc2"])

        # Verify they are gone
        results = store._search_vector([0.1, 0.2, 0.3, 0.4], top_k=4)
        result_ids = [r.id for r in results]
        assert "doc1" not in result_ids
        assert "doc2" not in result_ids
        assert len(results) == 2

    def test_delete_collection(self, dataset, temp_db_path, sample_documents):
        """Test deleting the entire collection."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Delete the collection
        store.delete()

        # Verify table is gone (search should fail or return empty)
        # Re-create to verify clean state
        store2 = DuckDBVector(dataset, database_path=temp_db_path)
        # This should work without errors (table doesn't exist yet)
        store2.create(sample_documents[:1])
        results = store2._search_vector([0.1, 0.2, 0.3, 0.4], top_k=4)
        assert len(results) == 1

    def test_empty_search_results(self, dataset, temp_db_path, sample_documents):
        """Test search with query that matches nothing."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        # Search for a term not in any document
        results = store.search("xyznonexistent", None, top_k=4, search_type="keyword")
        assert len(results) == 0

    def test_vector_search_without_vector(self, dataset, temp_db_path, sample_documents):
        """Test vector search with None query vector returns empty."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        results = store._search_vector(None, top_k=4)
        assert results == []

    def test_metadata_preserved(self, dataset, temp_db_path, sample_documents):
        """Test that document metadata is preserved through storage and retrieval."""
        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(sample_documents)

        results = store.search("python", None, top_k=1, search_type="keyword")
        assert len(results) == 1
        assert results[0].metadata.get("source") == "programming.txt"


class TestDuckDBFTSIntegration:
    """Integration tests for DuckDB FTS functionality."""

    def test_fts_with_stemming(self, dataset, temp_db_path):
        """Test that FTS properly stems words (e.g., 'learning' matches 'learn')."""
        docs = [
            Document(
                id="doc1",
                page_content="The student is learning machine learning concepts.",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={}
            ),
            Document(
                id="doc2",
                page_content="She learned to program in Python last year.",
                vector=[0.2, 0.3, 0.4, 0.5],
                metadata={}
            ),
        ]

        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(docs)

        # Search for "learn" should match both due to stemming
        results = store.search("learn", None, top_k=4, search_type="keyword")
        assert len(results) == 2

    def test_fts_case_insensitive(self, dataset, temp_db_path):
        """Test that FTS is case-insensitive."""
        docs = [
            Document(
                id="doc1",
                page_content="PYTHON is a PROGRAMMING language.",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={}
            ),
        ]

        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(docs)

        # Search with lowercase should match uppercase content
        results = store.search("python programming", None, top_k=4, search_type="keyword")
        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_hybrid_rrf_fusion(self, dataset, temp_db_path):
        """Test that hybrid search properly fuses vector and FTS results."""
        docs = [
            Document(
                id="doc1",
                page_content="Machine learning is transforming industries.",
                vector=[0.9, 0.1, 0.1, 0.1],  # Very different vector
                metadata={}
            ),
            Document(
                id="doc2",
                page_content="Cooking recipes for beginners.",
                vector=[0.1, 0.2, 0.3, 0.4],  # Similar vector to query
                metadata={}
            ),
        ]

        store = DuckDBVector(dataset, database_path=temp_db_path)
        store.create(docs)

        # Query: vector similar to doc2, but text about "machine learning" (doc1)
        # Hybrid should balance both signals
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = store.search("machine learning", query_vector, top_k=2, search_type="hybrid")

        assert len(results) == 2
        # Both documents should appear due to RRF fusion
        result_ids = {r.id for r in results}
        assert "doc1" in result_ids  # Matched by text
        assert "doc2" in result_ids  # Matched by vector
