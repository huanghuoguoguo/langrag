"""
Tests for Batch Processing functionality.

These tests verify:
1. BatchConfig validation
2. BatchProgress creation
3. BatchProcessor document processing
4. Progress callbacks
5. Error handling and retry logic
"""

from unittest.mock import MagicMock, patch, call
import time

import pytest

from langrag.batch import BatchConfig, BatchProcessor, BatchProgress
from langrag.batch.progress import BatchStage, LoggingProgressReporter
from langrag.entities.document import Document


class TestBatchConfig:
    """Tests for BatchConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BatchConfig()
        assert config.embedding_batch_size == 100
        assert config.storage_batch_size == 500
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.continue_on_error is False
        assert config.show_progress is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchConfig(
            embedding_batch_size=50,
            storage_batch_size=200,
            max_retries=5,
            retry_delay=2.0,
            continue_on_error=True,
            show_progress=False
        )
        assert config.embedding_batch_size == 50
        assert config.storage_batch_size == 200
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.continue_on_error is True
        assert config.show_progress is False

    def test_invalid_embedding_batch_size(self):
        """Test that invalid embedding_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="embedding_batch_size"):
            BatchConfig(embedding_batch_size=0)

    def test_invalid_storage_batch_size(self):
        """Test that invalid storage_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="storage_batch_size"):
            BatchConfig(storage_batch_size=0)

    def test_invalid_max_retries(self):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            BatchConfig(max_retries=-1)

    def test_invalid_retry_delay(self):
        """Test that negative retry_delay raises ValueError."""
        with pytest.raises(ValueError, match="retry_delay"):
            BatchConfig(retry_delay=-1.0)


class TestBatchProgress:
    """Tests for BatchProgress class."""

    def test_create_progress(self):
        """Test creating BatchProgress with auto-calculated percent."""
        progress = BatchProgress.create(
            stage=BatchStage.EMBEDDING,
            current=50,
            total=100,
            message="Test message"
        )

        assert progress.stage == BatchStage.EMBEDDING
        assert progress.current == 50
        assert progress.total == 100
        assert progress.percent == pytest.approx(0.5)
        assert progress.message == "Test message"

    def test_create_progress_zero_total(self):
        """Test creating BatchProgress with zero total (avoid division by zero)."""
        progress = BatchProgress.create(
            stage=BatchStage.COMPLETE,
            current=0,
            total=0
        )
        assert progress.percent == 0.0

    def test_progress_stages(self):
        """Test all batch stages are defined."""
        assert BatchStage.PARSING.value == "parsing"
        assert BatchStage.CHUNKING.value == "chunking"
        assert BatchStage.EMBEDDING.value == "embedding"
        assert BatchStage.STORING.value == "storing"
        assert BatchStage.COMPLETE.value == "complete"


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]] * 10
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.add_texts.return_value = None
        return store

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(id=f"doc{i}", page_content=f"Content {i}")
            for i in range(10)
        ]

    def test_process_empty_documents(self, mock_embedder, mock_vector_store):
        """Test processing empty document list."""
        processor = BatchProcessor(mock_embedder, mock_vector_store)
        stats = processor.process_documents([])

        assert stats["total"] == 0
        assert stats["embedded"] == 0
        assert stats["stored"] == 0
        assert stats["errors"] == 0
        mock_embedder.embed.assert_not_called()

    def test_process_documents_success(
        self, mock_embedder, mock_vector_store, sample_documents
    ):
        """Test successful document processing."""
        config = BatchConfig(embedding_batch_size=5, storage_batch_size=10)
        processor = BatchProcessor(mock_embedder, mock_vector_store, config)

        stats = processor.process_documents(sample_documents)

        assert stats["total"] == 10
        assert stats["embedded"] == 10
        assert stats["stored"] == 10
        assert stats["errors"] == 0
        assert stats["duration"] > 0

        # Should have called embed twice (5 + 5 documents)
        assert mock_embedder.embed.call_count == 2

        # Should have called add_texts once (all 10 in one storage batch)
        mock_vector_store.add_texts.assert_called_once()

    def test_process_with_progress_callback(
        self, mock_embedder, mock_vector_store, sample_documents
    ):
        """Test progress callback is called during processing."""
        config = BatchConfig(embedding_batch_size=5)
        processor = BatchProcessor(mock_embedder, mock_vector_store, config)

        progress_updates = []

        def on_progress(p: BatchProgress):
            progress_updates.append(p)

        processor.process_documents(sample_documents, on_progress=on_progress)

        # Should have multiple progress updates
        assert len(progress_updates) >= 2

        # Last update should be COMPLETE
        assert progress_updates[-1].stage == BatchStage.COMPLETE

    def test_process_with_embedding_error(
        self, mock_embedder, mock_vector_store, sample_documents
    ):
        """Test error handling during embedding."""
        mock_embedder.embed.side_effect = Exception("API Error")

        config = BatchConfig(max_retries=1, retry_delay=0.01)
        processor = BatchProcessor(mock_embedder, mock_vector_store, config)

        with pytest.raises(Exception, match="API Error"):
            processor.process_documents(sample_documents)

    def test_process_continue_on_error(
        self, mock_embedder, mock_vector_store, sample_documents
    ):
        """Test continue_on_error flag."""
        mock_embedder.embed.side_effect = Exception("API Error")

        config = BatchConfig(
            max_retries=0,
            continue_on_error=True,
            embedding_batch_size=5
        )
        processor = BatchProcessor(mock_embedder, mock_vector_store, config)

        stats = processor.process_documents(sample_documents)

        # All documents should be counted as errors
        assert stats["errors"] == 10
        assert stats["stored"] == 0

    def test_retry_logic(self, mock_embedder, mock_vector_store, sample_documents):
        """Test that retry logic works correctly."""
        # First two calls fail, third succeeds
        mock_embedder.embed.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            [[0.1, 0.2, 0.3]] * 10
        ]

        config = BatchConfig(
            max_retries=2,
            retry_delay=0.01,
            embedding_batch_size=10
        )
        processor = BatchProcessor(mock_embedder, mock_vector_store, config)

        stats = processor.process_documents(sample_documents)

        # Should succeed after retries
        assert stats["embedded"] == 10
        assert stats["errors"] == 0
        assert mock_embedder.embed.call_count == 3


class TestLoggingProgressReporter:
    """Tests for LoggingProgressReporter class."""

    def test_report_logs_progress(self):
        """Test that progress is logged correctly."""
        import logging

        reporter = LoggingProgressReporter("test.logger")
        progress = BatchProgress.create(
            stage=BatchStage.EMBEDDING,
            current=50,
            total=100
        )

        # Just verify it doesn't raise an error
        reporter.report(progress)
