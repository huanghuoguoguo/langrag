"""Tests for RetrievalService."""

from unittest.mock import MagicMock, patch

import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.service import RetrievalService


class TestRetrievalService:
    """Tests for RetrievalService class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        return Dataset(
            id="test-dataset",
            name="Test Dataset",
            description="Test description",
            collection_name="test_collection",
        )

    @pytest.fixture
    def mock_vector_manager(self):
        """Create a mock vector manager."""
        manager = MagicMock()
        manager.search.return_value = [
            Document(page_content="Result 1", metadata={"score": 0.9}),
            Document(page_content="Result 2", metadata={"score": 0.8}),
        ]
        return manager

    def test_retrieve_semantic_search(self, sample_dataset, mock_vector_manager):
        """Retrieve with semantic search method."""
        results = RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            retrieval_method="semantic_search",
            top_k=5,
            vector_manager=mock_vector_manager,
        )

        mock_vector_manager.search.assert_called_once_with(
            sample_dataset,
            "test query",
            None,  # query_vector
            top_k=5,
            search_type="similarity",
        )
        assert len(results) == 2

    def test_retrieve_hybrid_search(self, sample_dataset, mock_vector_manager):
        """Retrieve with hybrid search method."""
        results = RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            retrieval_method="hybrid_search",
            top_k=5,
            vector_manager=mock_vector_manager,
        )

        mock_vector_manager.search.assert_called_once_with(
            sample_dataset,
            "test query",
            None,
            top_k=5,
            search_type="hybrid",
        )
        assert len(results) == 2

    def test_retrieve_keyword_search_returns_empty(self, sample_dataset, mock_vector_manager):
        """Retrieve with keyword search returns empty (not implemented)."""
        results = RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            retrieval_method="keyword_search",
            top_k=5,
            vector_manager=mock_vector_manager,
        )

        # keyword_search is not fully implemented, should return empty
        assert results == []

    def test_retrieve_with_query_vector(self, sample_dataset, mock_vector_manager):
        """Retrieve with pre-computed query vector."""
        query_vector = [0.1, 0.2, 0.3]

        results = RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            query_vector=query_vector,
            retrieval_method="semantic_search",
            top_k=5,
            vector_manager=mock_vector_manager,
        )

        mock_vector_manager.search.assert_called_once_with(
            sample_dataset,
            "test query",
            query_vector,
            top_k=5,
            search_type="similarity",
        )

    @patch("langrag.datasource.vdb.global_manager.get_vector_manager")
    def test_retrieve_uses_global_manager_when_not_provided(
        self, mock_get_manager, sample_dataset
    ):
        """Uses global vector manager when not provided."""
        mock_global_manager = MagicMock()
        mock_global_manager.search.return_value = []
        mock_get_manager.return_value = mock_global_manager

        RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            retrieval_method="semantic_search",
            top_k=5,
            vector_manager=None,
        )

        mock_get_manager.assert_called_once()
        mock_global_manager.search.assert_called_once()

    def test_retrieve_default_retrieval_method(self, sample_dataset, mock_vector_manager):
        """Default retrieval method is semantic_search."""
        RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            top_k=5,
            vector_manager=mock_vector_manager,
        )

        mock_vector_manager.search.assert_called_once_with(
            sample_dataset,
            "test query",
            None,
            top_k=5,
            search_type="similarity",
        )

    def test_retrieve_default_top_k(self, sample_dataset, mock_vector_manager):
        """Default top_k is 4."""
        RetrievalService.retrieve(
            dataset=sample_dataset,
            query="test query",
            vector_manager=mock_vector_manager,
        )

        call_args = mock_vector_manager.search.call_args
        assert call_args.kwargs["top_k"] == 4
