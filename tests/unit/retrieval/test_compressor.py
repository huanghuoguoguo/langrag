"""Tests for compressor components."""

import pytest

from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.retrieval.compressor.providers.noop import NoOpCompressor


def _make_search_result(content: str, score: float = 0.9) -> SearchResult:
    """Helper to create a SearchResult."""
    doc = Document(page_content=content, metadata={})
    return SearchResult(chunk=doc, score=score)


class TestNoOpCompressor:
    """Tests for NoOpCompressor class."""

    def test_compress_returns_original(self):
        """NoOp compressor returns results unchanged."""
        compressor = NoOpCompressor()
        results = [
            _make_search_result("Content 1"),
            _make_search_result("Content 2"),
        ]

        compressed = compressor.compress("query", results)

        assert len(compressed) == 2
        assert compressed[0].chunk.page_content == "Content 1"
        assert compressed[1].chunk.page_content == "Content 2"

    def test_compress_empty_results(self):
        """NoOp compressor handles empty results."""
        compressor = NoOpCompressor()

        compressed = compressor.compress("query", [])

        assert compressed == []

    def test_compress_ignores_target_ratio(self):
        """NoOp compressor ignores target_ratio parameter."""
        compressor = NoOpCompressor()
        results = [_make_search_result("Content")]

        # Should work with any target_ratio
        compressed1 = compressor.compress("query", results, target_ratio=0.1)
        compressed2 = compressor.compress("query", results, target_ratio=0.9)

        assert len(compressed1) == 1
        assert len(compressed2) == 1

    def test_compress_preserves_scores(self):
        """NoOp compressor preserves original scores."""
        compressor = NoOpCompressor()
        results = [
            _make_search_result("Content 1", score=0.95),
            _make_search_result("Content 2", score=0.85),
        ]

        compressed = compressor.compress("query", results)

        assert compressed[0].score == 0.95
        assert compressed[1].score == 0.85
