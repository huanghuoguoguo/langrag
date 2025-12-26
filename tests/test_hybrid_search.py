"""Integration tests for hybrid search and RRF fusion.

This test suite demonstrates:
1. Vector store capability detection
2. RRF fusion combining multiple search results
3. Weighted RRF for custom ranking
4. SeekDB hybrid search (requires pyseekdb)
"""

import pytest
from pathlib import Path
import tempfile

from langrag import (
    Document,
    Chunk,
    SearchResult,
    VectorStoreCapabilities,
    SearchMode,
    InMemoryVectorStore,
    VectorStoreFactory,
    reciprocal_rank_fusion,
    weighted_rrf
)


class TestVectorStoreCapabilities:
    """Test capability detection for vector stores."""

    def test_in_memory_capabilities(self):
        """InMemoryVectorStore should only support vector search."""
        store = InMemoryVectorStore()
        caps = store.capabilities

        assert caps.supports_vector is True
        assert caps.supports_fulltext is False
        assert caps.supports_hybrid is False

    def test_capability_validation(self):
        """Capability validation should raise for unsupported modes."""
        caps = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=False,
            supports_hybrid=False
        )

        # Should not raise for supported mode
        caps.validate_mode(SearchMode.VECTOR)

        # Should raise for unsupported modes
        with pytest.raises(ValueError, match="Full-text search not supported"):
            caps.validate_mode(SearchMode.FULLTEXT)

        with pytest.raises(ValueError, match="Hybrid search not supported"):
            caps.validate_mode(SearchMode.HYBRID)

    def test_unsupported_search_methods(self):
        """InMemoryVectorStore should raise for unsupported search methods."""
        store = InMemoryVectorStore()

        # Add some test data
        chunks = [
            Chunk(
                content="Test content",
                embedding=[0.1, 0.2, 0.3],
                source_doc_id="doc1",
                metadata={}
            )
        ]
        store.add(chunks)

        # Full-text search should raise
        with pytest.raises(NotImplementedError, match="does not support full-text search"):
            store.search_fulltext("test query")

        # Hybrid search should raise
        with pytest.raises(NotImplementedError, match="does not support native hybrid search"):
            store.search_hybrid(
                query_vector=[0.1, 0.2, 0.3],
                query_text="test query"
            )


class TestRRFFusion:
    """Test Reciprocal Rank Fusion for combining results."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(id="chunk1", content="Python programming", embedding=[1.0, 0.0, 0.0], source_doc_id="doc1"),
            Chunk(id="chunk2", content="Java development", embedding=[0.0, 1.0, 0.0], source_doc_id="doc1"),
            Chunk(id="chunk3", content="Python data science", embedding=[0.8, 0.2, 0.0], source_doc_id="doc2"),
            Chunk(id="chunk4", content="Machine learning", embedding=[0.0, 0.0, 1.0], source_doc_id="doc2"),
            Chunk(id="chunk5", content="Python web framework", embedding=[0.9, 0.0, 0.1], source_doc_id="doc3"),
        ]

    def test_basic_rrf(self, sample_chunks):
        """Test basic RRF fusion combining two result lists."""
        # Simulate two different search results
        list1 = [
            SearchResult(chunk=sample_chunks[0], score=0.95),
            SearchResult(chunk=sample_chunks[2], score=0.85),
            SearchResult(chunk=sample_chunks[4], score=0.75),
        ]

        list2 = [
            SearchResult(chunk=sample_chunks[2], score=0.90),  # chunk3 appears in both
            SearchResult(chunk=sample_chunks[1], score=0.80),
            SearchResult(chunk=sample_chunks[3], score=0.70),
        ]

        # Fuse results
        fused = reciprocal_rank_fusion([list1, list2], top_k=5)

        # Verify chunk3 ranks high (appears in both lists)
        assert len(fused) <= 5
        assert any(r.chunk.id == "chunk3" for r in fused)

        # Chunk3 should rank higher than chunks appearing in only one list
        chunk3_rank = next(i for i, r in enumerate(fused) if r.chunk.id == "chunk3")
        assert chunk3_rank == 0  # Should be first due to appearing in both lists

    def test_rrf_with_empty_lists(self):
        """RRF should handle empty input gracefully."""
        # Empty list
        assert reciprocal_rank_fusion([]) == []

        # Lists with empty sublists
        list1 = []
        list2 = [SearchResult(
            chunk=Chunk(id="c1", content="test", embedding=[1.0], source_doc_id="doc1"),
            score=0.9
        )]
        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) == 1

    def test_weighted_rrf(self, sample_chunks):
        """Test weighted RRF with custom weights."""
        list1 = [
            SearchResult(chunk=sample_chunks[0], score=0.95),
            SearchResult(chunk=sample_chunks[1], score=0.85),
        ]

        list2 = [
            SearchResult(chunk=sample_chunks[2], score=0.90),
            SearchResult(chunk=sample_chunks[3], score=0.80),
        ]

        # Favor first list heavily (90% vs 10%)
        fused = weighted_rrf([list1, list2], weights=[0.9, 0.1], top_k=4)

        assert len(fused) <= 4
        # Chunks from list1 should generally rank higher
        top_chunk_id = fused[0].chunk.id
        assert top_chunk_id in ["chunk1", "chunk2"]  # From list1

    def test_weighted_rrf_validation(self, sample_chunks):
        """Weighted RRF should validate input."""
        list1 = [SearchResult(chunk=sample_chunks[0], score=0.9)]

        # Mismatched weights length
        with pytest.raises(ValueError, match="Weights length.*must match"):
            weighted_rrf([list1], weights=[0.5, 0.5])

        # Zero sum weights
        with pytest.raises(ValueError, match="Sum of weights must be > 0"):
            weighted_rrf([list1], weights=[0.0])

    def test_rrf_score_calculation(self):
        """Verify RRF score calculation formula."""
        chunk = Chunk(id="c1", content="test", embedding=[1.0], source_doc_id="doc1")

        # Single list, single result
        list1 = [SearchResult(chunk=chunk, score=1.0)]
        fused = reciprocal_rank_fusion([list1], k=60)

        # RRF score should be 1/(60 + 1) = 1/61 ≈ 0.01639
        assert len(fused) == 1
        assert abs(fused[0].score - 1/61) < 0.0001

        # Same chunk in two lists
        list2 = [SearchResult(chunk=chunk, score=0.9)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        # RRF score should be 1/61 + 1/61 = 2/61 ≈ 0.03279
        assert len(fused) == 1
        assert abs(fused[0].score - 2/61) < 0.0001


class TestMultiStoreHybridSearch:
    """Test hybrid search using multiple stores with RRF."""

    def test_simulated_hybrid_with_rrf(self):
        """Simulate hybrid search: vector store + text store + RRF."""
        # Create two stores
        vector_store = InMemoryVectorStore()

        # Add chunks
        chunks = [
            Chunk(
                id="c1",
                content="Python is a programming language",
                embedding=[1.0, 0.0, 0.0],
                source_doc_id="doc1",
                metadata={"topic": "programming"}
            ),
            Chunk(
                id="c2",
                content="Java is also a programming language",
                embedding=[0.9, 0.1, 0.0],
                source_doc_id="doc1",
                metadata={"topic": "programming"}
            ),
            Chunk(
                id="c3",
                content="Python is popular for data science",
                embedding=[0.8, 0.0, 0.2],
                source_doc_id="doc2",
                metadata={"topic": "data science"}
            ),
        ]
        vector_store.add(chunks)

        # Simulate vector search results
        query_vec = [0.95, 0.05, 0.0]
        vector_results = vector_store.search(query_vec, top_k=3)

        # Simulate text search results (in practice, this would come from a text store)
        # For this test, we manually create results favoring "Python" mentions
        text_results = [
            SearchResult(chunk=chunks[0], score=0.9),  # Python mentioned
            SearchResult(chunk=chunks[2], score=0.85),  # Python mentioned
            SearchResult(chunk=chunks[1], score=0.3),   # No Python
        ]

        # Fuse with RRF
        hybrid_results = reciprocal_rank_fusion(
            [vector_results, text_results],
            top_k=3
        )

        # Should have at most 3 results
        assert len(hybrid_results) <= 3

        # Chunks mentioned in both lists should rank higher
        # (exact order depends on scores, but c1 and c3 likely top 2)
        top_ids = [r.chunk.id for r in hybrid_results[:2]]
        assert "c1" in top_ids or "c3" in top_ids


class TestSeekDBVectorStore:
    """Test SeekDB vector store (requires pyseekdb)."""

    @staticmethod
    def seekdb_available():
        """Check if pyseekdb is actually importable."""
        try:
            import pyseekdb
            return True
        except ImportError:
            return False

    def test_seekdb_registration(self):
        """SeekDB should be registered if pyseekdb is available."""
        available_types = VectorStoreFactory.list_types()

        # In-memory should always be available
        assert "in_memory" in available_types

        # SeekDB may or may not be available
        # (depends on whether pyseekdb is installed)
        if self.seekdb_available():
            assert "seekdb" in available_types

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB not available (pyseekdb not installed)"
    )
    def test_seekdb_capabilities(self):
        """SeekDB should support vector search (current pyseekdb client limitations)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStoreFactory.create(
                "seekdb",
                collection_name="test_collection",
                dimension=3,
                mode="embedded",
                db_path=tmpdir
            )

            caps = store.capabilities
            assert caps.supports_vector is True
            assert caps.supports_fulltext is False  # Not supported by current pyseekdb client
            assert caps.supports_hybrid is False    # Not supported by current pyseekdb client

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB not available (pyseekdb not installed)"
    )
    def test_seekdb_vector_search(self):
        """Test SeekDB's vector search capabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStoreFactory.create(
                "seekdb",
                collection_name="test_vector",
                dimension=3,
                mode="embedded",
                db_path=tmpdir
            )

            # Add test data
            chunks = [
                Chunk(
                    id="v1",
                    content="Python programming language",
                    embedding=[1.0, 0.0, 0.0],
                    source_doc_id="doc1",
                    metadata={"lang": "python"}
                ),
                Chunk(
                    id="v2",
                    content="Machine learning with Python",
                    embedding=[0.8, 0.2, 0.0],
                    source_doc_id="doc2",
                    metadata={"lang": "python"}
                ),
            ]
            store.add(chunks)

            # Perform vector search
            results = store.search(
                query_vector=[0.9, 0.1, 0.0],
                top_k=2
            )

            # Should return results
            assert len(results) > 0
            assert len(results) <= 2

            # Results should be SearchResult instances
            assert all(isinstance(r, SearchResult) for r in results)

            # Check that results have proper chunk data
            for result in results:
                assert result.chunk.id in ["v1", "v2"]
                assert result.score >= 0.0
                assert result.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
