"""Tests for ParagraphIndexProcessor."""

from unittest.mock import MagicMock, patch

import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.paragraph import ParagraphIndexProcessor


class TestParagraphIndexProcessor:
    """Tests for ParagraphIndexProcessor class."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        return embedder

    @pytest.fixture
    def mock_vector_manager(self):
        """Create a mock vector manager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def mock_splitter(self):
        """Create a mock splitter."""
        splitter = MagicMock()
        # Splitter returns the same documents (no split)
        splitter.split_documents.side_effect = lambda docs: docs
        return splitter

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
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(page_content="First document content", metadata={"document_id": "doc1"}),
            Document(page_content="Second document content", metadata={"document_id": "doc2"}),
        ]

    def test_init_with_all_components(self, mock_embedder, mock_vector_manager, mock_splitter):
        """Initialize with all components."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
            splitter=mock_splitter,
        )

        assert processor.embedder == mock_embedder
        assert processor.vector_manager == mock_vector_manager
        assert processor.splitter == mock_splitter
        assert processor.cleaner is not None

    def test_init_default_cleaner(self, mock_embedder):
        """Default cleaner is created if not provided."""
        processor = ParagraphIndexProcessor(embedder=mock_embedder)

        assert processor.cleaner is not None

    def test_process_with_splitter(
        self, mock_embedder, mock_vector_manager, mock_splitter, sample_dataset, sample_documents
    ):
        """Process documents with splitter."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
            splitter=mock_splitter,
        )

        processor.process(sample_dataset, sample_documents)

        # Splitter should be called for each document
        assert mock_splitter.split_documents.call_count == 2
        # Embedder should be called once with all texts
        mock_embedder.embed_documents.assert_called_once()
        # Vector manager should save the chunks
        mock_vector_manager.add_texts.assert_called_once()

    def test_process_without_splitter(
        self, mock_embedder, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Process documents without splitter treats whole doc as chunk."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
            splitter=None,
        )

        processor.process(sample_dataset, sample_documents)

        # Embedder should be called with document contents
        mock_embedder.embed_documents.assert_called_once()
        # Vector manager should save the documents
        mock_vector_manager.add_texts.assert_called_once()

    def test_process_empty_documents(self, mock_embedder, mock_vector_manager, sample_dataset):
        """Process empty document list."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, [])

        # Should not call embedder or vector manager
        mock_embedder.embed_documents.assert_not_called()
        mock_vector_manager.add_texts.assert_not_called()

    def test_process_adds_dataset_id_to_metadata(
        self, mock_embedder, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Dataset ID is added to chunk metadata."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, sample_documents)

        # Check that add_texts was called with chunks containing dataset_id
        call_args = mock_vector_manager.add_texts.call_args
        chunks = call_args[0][1]  # Second positional argument
        for chunk in chunks:
            assert chunk.metadata["dataset_id"] == "test-dataset"

    def test_process_preserves_document_id(
        self, mock_embedder, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Document ID is preserved in chunk metadata."""
        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, sample_documents)

        call_args = mock_vector_manager.add_texts.call_args
        chunks = call_args[0][1]

        assert chunks[0].metadata["document_id"] == "doc1"
        assert chunks[1].metadata["document_id"] == "doc2"

    def test_process_sets_vectors(
        self, mock_embedder, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Vectors are set on chunks after embedding."""
        mock_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, sample_documents)

        call_args = mock_vector_manager.add_texts.call_args
        chunks = call_args[0][1]

        assert chunks[0].vector == [0.1, 0.2]
        assert chunks[1].vector == [0.3, 0.4]

    def test_process_cleans_content(
        self, mock_embedder, mock_vector_manager, sample_dataset
    ):
        """Content is cleaned before processing."""
        docs = [Document(page_content="  Content with spaces  ", metadata={})]

        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, docs)

        # Cleaner should have modified the content
        mock_embedder.embed_documents.assert_called_once()

    @patch("langrag.datasource.vdb.global_manager.get_vector_manager")
    def test_process_uses_global_manager_when_not_provided(
        self, mock_get_manager, mock_embedder, sample_dataset, sample_documents
    ):
        """Uses global vector manager when not provided."""
        mock_global_manager = MagicMock()
        mock_get_manager.return_value = mock_global_manager

        processor = ParagraphIndexProcessor(
            embedder=mock_embedder,
            vector_manager=None,
        )

        processor.process(sample_dataset, sample_documents)

        mock_get_manager.assert_called_once()
        mock_global_manager.add_texts.assert_called_once()
