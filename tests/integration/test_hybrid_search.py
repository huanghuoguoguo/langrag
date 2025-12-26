"""Integration tests for hybrid search with multiple stores.

集成测试特点：
- 测试多存储协作
- 验证RRF融合实际效果
- 测试SeekDB混合搜索
"""

import pytest
from pathlib import Path
import tempfile

from langrag import (
    Chunk,
    SearchResult,
    InMemoryVectorStore,
    VectorStoreFactory,
    reciprocal_rank_fusion
)


@pytest.mark.integration
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
