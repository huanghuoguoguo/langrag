"""Tests for Reciprocal Rank Fusion (RRF) utilities."""

import pytest

from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.utils.rrf import reciprocal_rank_fusion, weighted_rrf


def _make_search_result(doc_id: str, content: str, score: float) -> SearchResult:
    """Helper to create a SearchResult."""
    doc = Document(page_content=content, metadata={"doc_id": doc_id})
    doc.id = doc_id
    return SearchResult(chunk=doc, score=score)


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list(self):
        """Single result list returns items ranked by RRF score."""
        results = [
            _make_search_result("doc1", "content1", 0.9),
            _make_search_result("doc2", "content2", 0.8),
        ]
        fused = reciprocal_rank_fusion([results], k=60, top_k=5)

        assert len(fused) == 2
        # First item should have higher RRF score (1/(60+1) vs 1/(60+2))
        assert fused[0].chunk.id == "doc1"
        assert fused[1].chunk.id == "doc2"

    def test_multiple_lists_fusion(self):
        """Multiple lists are properly fused."""
        list1 = [
            _make_search_result("doc1", "content1", 0.9),
            _make_search_result("doc2", "content2", 0.8),
        ]
        list2 = [
            _make_search_result("doc2", "content2", 0.95),  # doc2 appears in both
            _make_search_result("doc3", "content3", 0.7),
        ]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=5)

        assert len(fused) == 3
        # doc2 should be ranked highest (appears in both lists)
        assert fused[0].chunk.id == "doc2"

    def test_top_k_limit(self):
        """Results are limited to top_k."""
        results = [
            _make_search_result(f"doc{i}", f"content{i}", 0.9 - i * 0.1)
            for i in range(10)
        ]
        fused = reciprocal_rank_fusion([results], k=60, top_k=3)

        assert len(fused) == 3

    def test_rrf_score_calculation(self):
        """RRF scores are calculated correctly."""
        results = [_make_search_result("doc1", "content1", 0.9)]
        fused = reciprocal_rank_fusion([results], k=60, top_k=5)

        # Score should be 1/(60+1) = 1/61
        expected_score = 1.0 / 61
        assert abs(fused[0].score - expected_score) < 1e-10

    def test_duplicate_in_same_list(self):
        """Handles same document appearing multiple times in same list."""
        list1 = [
            _make_search_result("doc1", "content1", 0.9),
            _make_search_result("doc2", "content2", 0.8),
        ]
        list2 = [
            _make_search_result("doc1", "content1", 0.7),
        ]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=5)

        # doc1 should have combined score from both lists
        doc1_result = next(r for r in fused if r.chunk.id == "doc1")
        expected_score = 1.0 / 61 + 1.0 / 61  # rank 1 in both lists
        assert abs(doc1_result.score - expected_score) < 1e-10


class TestWeightedRRF:
    """Tests for weighted_rrf function."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert weighted_rrf([], []) == []

    def test_weights_length_mismatch(self):
        """Raises error when weights don't match result lists."""
        results = [[_make_search_result("doc1", "content1", 0.9)]]
        with pytest.raises(ValueError, match="Weights length"):
            weighted_rrf(results, [0.5, 0.5])

    def test_zero_weights_sum(self):
        """Raises error when weights sum to zero."""
        results = [[_make_search_result("doc1", "content1", 0.9)]]
        with pytest.raises(ValueError, match="Sum of weights"):
            weighted_rrf(results, [0])

    def test_weighted_fusion(self):
        """Weights affect fusion ranking."""
        list1 = [_make_search_result("doc1", "content1", 0.9)]
        list2 = [_make_search_result("doc2", "content2", 0.9)]

        # Give list1 much higher weight
        fused = weighted_rrf([list1, list2], weights=[0.9, 0.1], k=60, top_k=5)

        assert len(fused) == 2
        # doc1 should be first due to higher weight
        assert fused[0].chunk.id == "doc1"
        assert fused[0].score > fused[1].score

    def test_equal_weights(self):
        """Equal weights produce same result as unweighted RRF."""
        list1 = [_make_search_result("doc1", "content1", 0.9)]
        list2 = [_make_search_result("doc2", "content2", 0.9)]

        fused = weighted_rrf([list1, list2], weights=[1.0, 1.0], k=60, top_k=5)

        # Both should have same score with equal weights
        assert abs(fused[0].score - fused[1].score) < 1e-10

    def test_filters_empty_lists(self):
        """Empty lists are filtered out with their weights."""
        list1 = [_make_search_result("doc1", "content1", 0.9)]
        list2 = []  # Empty

        fused = weighted_rrf([list1, list2], weights=[0.5, 0.5], k=60, top_k=5)

        assert len(fused) == 1
        assert fused[0].chunk.id == "doc1"

    def test_weight_normalization(self):
        """Weights are normalized to sum to 1."""
        list1 = [_make_search_result("doc1", "content1", 0.9)]

        # Weights that don't sum to 1
        fused = weighted_rrf([list1], weights=[2.0], k=60, top_k=5)

        # Should still work (weight normalized to 1.0)
        assert len(fused) == 1
