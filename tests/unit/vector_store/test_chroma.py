"""Test Chroma vector store."""

import tempfile

import pytest

from langrag import (
    Chunk,
    SearchResult,
    VectorStoreFactory,
)


class TestChromaVectorStore:
    """Test Chroma vector store functionality."""

    @staticmethod
    def chroma_available():
        """Check if chromadb is available."""
        import importlib.util

        return importlib.util.find_spec("chromadb") is not None

    def test_chroma_registration(self):
        """Chroma should be registered if chromadb is available."""
        available = VectorStoreFactory.list_types()

        # InMemory should always be available
        assert "in_memory" in available

        # Chroma may or may not be available
        if self.chroma_available():
            assert "chroma" in available
        else:
            assert "chroma" not in available

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_ephemeral_mode(self):
        """Test Chroma in ephemeral (in-memory) mode."""
        store = VectorStoreFactory.create("chroma", collection_name="test_ephemeral")

        # Check capabilities
        caps = store.capabilities
        assert caps.supports_vector is True
        assert caps.supports_fulltext is False
        assert caps.supports_hybrid is False

        # Should start empty
        assert store.count() == 0

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_persistent_mode(self):
        """Test Chroma in persistent mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStoreFactory.create(
                "chroma", collection_name="test_persistent", persist_directory=tmpdir
            )

            # Add some data
            chunks = [
                Chunk(
                    id="c1",
                    content="Test content 1",
                    embedding=[1.0, 0.0, 0.0],
                    source_doc_id="doc1",
                    metadata={"index": 0},
                ),
                Chunk(
                    id="c2",
                    content="Test content 2",
                    embedding=[0.0, 1.0, 0.0],
                    source_doc_id="doc1",
                    metadata={"index": 1},
                ),
            ]
            store.add(chunks)
            assert store.count() == 2

            # Create new store instance with same persist_directory
            store2 = VectorStoreFactory.create(
                "chroma", collection_name="test_persistent", persist_directory=tmpdir
            )

            # Should load existing data
            assert store2.count() == 2

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_add_and_search(self):
        """Test adding chunks and searching."""
        store = VectorStoreFactory.create("chroma", collection_name="test_add_search")

        # Add chunks
        chunks = [
            Chunk(
                id="c1",
                content="Python programming language",
                embedding=[1.0, 0.0, 0.0],
                source_doc_id="doc1",
                metadata={"topic": "programming"},
            ),
            Chunk(
                id="c2",
                content="Java development guide",
                embedding=[0.9, 0.1, 0.0],
                source_doc_id="doc2",
                metadata={"topic": "programming"},
            ),
            Chunk(
                id="c3",
                content="Machine learning basics",
                embedding=[0.0, 0.0, 1.0],
                source_doc_id="doc3",
                metadata={"topic": "ml"},
            ),
        ]
        store.add(chunks)
        assert store.count() == 3

        # Search with vector similar to c1
        query_vector = [0.95, 0.05, 0.0]
        results = store.search(query_vector, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

        # First result should be c1 (most similar)
        assert results[0].chunk.id == "c1"
        assert results[0].score > results[1].score

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_metadata_filter(self):
        """Test metadata filtering in search."""
        store = VectorStoreFactory.create("chroma", collection_name="test_metadata_filter")

        # Add chunks with different metadata
        chunks = [
            Chunk(
                id="prog1",
                content="Python tutorial",
                embedding=[1.0, 0.0, 0.0],
                source_doc_id="doc1",
                metadata={"category": "programming", "language": "python"},
            ),
            Chunk(
                id="prog2",
                content="Java tutorial",
                embedding=[0.9, 0.1, 0.0],
                source_doc_id="doc2",
                metadata={"category": "programming", "language": "java"},
            ),
            Chunk(
                id="ml1",
                content="ML with Python",
                embedding=[0.8, 0.0, 0.2],
                source_doc_id="doc3",
                metadata={"category": "ml", "language": "python"},
            ),
        ]
        store.add(chunks)

        # Search only for programming category
        query_vector = [1.0, 0.0, 0.0]
        results = store.search(query_vector, top_k=10, metadata_filter={"category": "programming"})

        # Should only get prog1 and prog2
        assert len(results) == 2
        result_ids = {r.chunk.id for r in results}
        assert result_ids == {"prog1", "prog2"}

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_delete(self):
        """Test deleting chunks."""
        store = VectorStoreFactory.create("chroma", collection_name="test_delete")

        # Add chunks
        chunks = [
            Chunk(
                id=f"c{i}",
                content=f"Content {i}",
                embedding=[float(i), 0.0, 0.0],
                source_doc_id="doc1",
                metadata={},
            )
            for i in range(5)
        ]
        store.add(chunks)
        assert store.count() == 5

        # Delete some chunks
        store.delete(["c1", "c3"])
        assert store.count() == 3

        # Verify deleted chunks are gone
        results = store.search([1.0, 0.0, 0.0], top_k=10)
        result_ids = {r.chunk.id for r in results}
        assert "c1" not in result_ids
        assert "c3" not in result_ids

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_clear(self):
        """Test clearing all chunks."""
        store = VectorStoreFactory.create("chroma", collection_name="test_clear")

        # Add chunks
        chunks = [
            Chunk(
                id=f"c{i}",
                content=f"Content {i}",
                embedding=[float(i), 0.0, 0.0],
                source_doc_id="doc1",
                metadata={},
            )
            for i in range(3)
        ]
        store.add(chunks)
        assert store.count() == 3

        # Clear all
        store.clear()
        assert store.count() == 0

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_distance_metrics(self):
        """Test different distance metrics."""
        metrics = ["cosine", "l2", "ip"]

        for metric in metrics:
            store = VectorStoreFactory.create(
                "chroma", collection_name=f"test_{metric}", distance_metric=metric
            )

            # Add test data
            chunks = [
                Chunk(
                    id="c1",
                    content="Test 1",
                    embedding=[1.0, 0.0, 0.0],
                    source_doc_id="doc1",
                    metadata={},
                ),
                Chunk(
                    id="c2",
                    content="Test 2",
                    embedding=[0.0, 1.0, 0.0],
                    source_doc_id="doc1",
                    metadata={},
                ),
            ]
            store.add(chunks)

            # Search should work
            results = store.search([1.0, 0.0, 0.0], top_k=2)
            assert len(results) == 2
            assert all(r.score >= 0 for r in results)

    @pytest.mark.skipif(
        not chroma_available.__func__(), reason="Chroma not available (chromadb not installed)"
    )
    def test_chroma_missing_embedding_error(self):
        """Test that missing embeddings raise ValueError."""
        store = VectorStoreFactory.create("chroma", collection_name="test_missing_embedding")

        # Try to add chunk without embedding
        chunk = Chunk(
            id="c1",
            content="Test",
            embedding=None,  # Missing!
            source_doc_id="doc1",
            metadata={},
        )

        with pytest.raises(ValueError, match="missing embedding"):
            store.add([chunk])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
