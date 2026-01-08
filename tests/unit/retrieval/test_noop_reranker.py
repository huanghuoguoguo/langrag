"""Tests for NoOpReranker."""

import pytest

from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.retrieval.rerank.providers.noop import NoOpReranker


def _make_search_result(content: str, score: float = 0.9) -> SearchResult:
    """Helper to create a SearchResult."""
    doc = Document(page_content=content, metadata={})
    return SearchResult(chunk=doc, score=score)


class TestNoOpReranker:
    """Tests for NoOpReranker class."""

    def test_rerank_returns_original(self):
        """NoOp reranker returns results unchanged."""
        reranker = NoOpReranker()
        results = [
            _make_search_result("Content 1", 0.9),
            _make_search_result("Content 2", 0.8),
        ]

        reranked = reranker.rerank("query", results)

        assert len(reranked) == 2
        assert reranked[0].chunk.page_content == "Content 1"
        assert reranked[1].chunk.page_content == "Content 2"

    def test_rerank_empty_results(self):
        """NoOp reranker handles empty results."""
        reranker = NoOpReranker()

        reranked = reranker.rerank("query", [])

        assert reranked == []

    def test_rerank_with_top_k(self):
        """NoOp reranker respects top_k parameter."""
        reranker = NoOpReranker()
        results = [
            _make_search_result("Content 1"),
            _make_search_result("Content 2"),
            _make_search_result("Content 3"),
        ]

        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].chunk.page_content == "Content 1"
        assert reranked[1].chunk.page_content == "Content 2"

    def test_rerank_top_k_none(self):
        """NoOp reranker returns all when top_k is None."""
        reranker = NoOpReranker()
        results = [_make_search_result(f"Content {i}") for i in range(5)]

        reranked = reranker.rerank("query", results, top_k=None)

        assert len(reranked) == 5

    def test_rerank_preserves_scores(self):
        """NoOp reranker preserves original scores."""
        reranker = NoOpReranker()
        results = [
            _make_search_result("Content 1", score=0.95),
            _make_search_result("Content 2", score=0.85),
        ]

        reranked = reranker.rerank("query", results)

        assert reranked[0].score == 0.95
        assert reranked[1].score == 0.85

    def test_rerank_top_k_larger_than_results(self):
        """top_k larger than results returns all results."""
        reranker = NoOpReranker()
        results = [_make_search_result("Content 1")]

        reranked = reranker.rerank("query", results, top_k=10)

        assert len(reranked) == 1
