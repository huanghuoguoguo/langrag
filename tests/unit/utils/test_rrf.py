"""Unit tests for RRF (Reciprocal Rank Fusion) algorithms.

单元测试特点：
- 快速执行
- 测试RRF算法逻辑
- 验证边界情况
"""

import pytest
from langrag import Chunk, SearchResult, reciprocal_rank_fusion, weighted_rrf


@pytest.mark.unit
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


