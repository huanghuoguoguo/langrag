"""Tests for RAPTOR Index Processor."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.raptor import (
    RaptorConfig,
    RaptorProcessor,
    RaptorIndexProcessor,
)


class TestRaptorConfig:
    """Tests for RaptorConfig dataclass."""

    def test_default_values(self):
        """Default configuration values are set correctly."""
        config = RaptorConfig()

        assert config.max_cluster == 64
        assert config.threshold == 0.1
        assert config.max_token == 512
        assert config.max_errors == 3
        assert config.random_state == 42
        assert config.umap_n_components == 12
        assert config.umap_metric == "cosine"
        assert "{cluster_content}" in config.summarize_prompt

    def test_custom_values(self):
        """Custom configuration values are accepted."""
        config = RaptorConfig(
            max_cluster=32,
            threshold=0.2,
            max_token=256,
            max_errors=5,
            random_state=123,
        )

        assert config.max_cluster == 32
        assert config.threshold == 0.2
        assert config.max_token == 256
        assert config.max_errors == 5
        assert config.random_state == 123


class TestRaptorProcessor:
    """Tests for RaptorProcessor core algorithm."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = "This is a summary of the cluster content."
        llm.embed_query.return_value = [0.1, 0.2, 0.3]
        return llm

    @pytest.fixture
    def processor(self, mock_llm):
        """Create a RaptorProcessor instance."""
        return RaptorProcessor(llm=mock_llm)

    def test_init(self, mock_llm):
        """Initialize processor with LLM."""
        processor = RaptorProcessor(llm=mock_llm)

        assert processor._llm == mock_llm
        assert processor._config is not None
        assert processor._error_count == 0

    def test_init_with_custom_config(self, mock_llm):
        """Initialize processor with custom config."""
        config = RaptorConfig(max_cluster=32)
        processor = RaptorProcessor(llm=mock_llm, config=config)

        assert processor._config.max_cluster == 32

    def test_get_optimal_clusters(self, processor):
        """Get optimal number of clusters using BIC."""
        # Create sample embeddings (5 points in 3D)
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.8, 0.9, 1.0],
            [0.85, 0.95, 1.05],
            [0.5, 0.5, 0.5],
        ])

        n_clusters = processor._get_optimal_clusters(embeddings, random_state=42)

        assert isinstance(n_clusters, int)
        assert 1 <= n_clusters <= 5

    @pytest.mark.asyncio
    async def test_summarize_cluster(self, processor, mock_llm):
        """Summarize a cluster of texts."""
        texts = ["First chunk of text.", "Second chunk of text."]

        result = await processor._summarize_cluster(texts)

        assert result is not None
        summary, embedding = result
        assert summary == "This is a summary of the cluster content."
        assert embedding == [0.1, 0.2, 0.3]
        mock_llm.chat.assert_called_once()
        mock_llm.embed_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_cluster_error_handling(self, mock_llm):
        """Handle errors during summarization."""
        mock_llm.chat.side_effect = Exception("LLM error")
        processor = RaptorProcessor(llm=mock_llm, config=RaptorConfig(max_errors=2))

        result = await processor._summarize_cluster(["Text"])

        assert result is None
        assert processor._error_count == 1

    @pytest.mark.asyncio
    async def test_summarize_cluster_max_errors(self, mock_llm):
        """Abort after max errors."""
        mock_llm.chat.side_effect = Exception("LLM error")
        config = RaptorConfig(max_errors=2)
        processor = RaptorProcessor(llm=mock_llm, config=config)

        # First call increments error count but doesn't raise
        result = await processor._summarize_cluster(["Text"])
        assert result is None
        assert processor._error_count == 1

        # Second call should raise because error_count reaches max_errors
        with pytest.raises(RuntimeError, match="RAPTOR aborted"):
            await processor._summarize_cluster(["Text"])

    @pytest.mark.asyncio
    async def test_process_single_chunk(self, processor):
        """Process single chunk returns it unchanged."""
        chunks = [("Single chunk", [0.1, 0.2, 0.3])]

        result = await processor.process(chunks)

        assert len(result) == 1
        assert result[0][0] == "Single chunk"
        assert result[0][1] == [0.1, 0.2, 0.3]
        assert result[0][2] == 0  # Layer 0

    @pytest.mark.asyncio
    async def test_process_empty_chunks(self, processor):
        """Process empty chunks returns empty list."""
        result = await processor.process([])

        assert result == []

    @pytest.mark.asyncio
    async def test_process_filters_invalid_chunks(self, processor):
        """Filter out invalid chunks."""
        chunks = [
            ("Valid chunk", [0.1, 0.2]),
            ("", [0.1, 0.2]),  # Empty text
            ("Another valid", None),  # None embedding
            ("Valid", []),  # Empty embedding
        ]

        result = await processor.process(chunks)

        # Only first chunk is valid
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_process_two_chunks(self, processor, mock_llm):
        """Process two chunks creates one summary."""
        chunks = [
            ("First chunk", [0.1, 0.2, 0.3]),
            ("Second chunk", [0.4, 0.5, 0.6]),
        ]

        result = await processor.process(chunks)

        # 2 original + 1 summary
        assert len(result) >= 2
        # Original chunks should be layer 0
        assert result[0][2] == 0
        assert result[1][2] == 0

    @pytest.mark.asyncio
    async def test_process_callback(self, processor, mock_llm):
        """Callback is called during processing."""
        chunks = [
            ("First chunk", [0.1, 0.2, 0.3]),
            ("Second chunk", [0.4, 0.5, 0.6]),
        ]
        callback = MagicMock()

        await processor.process(chunks, callback=callback)

        # Callback should have been called at least once
        assert callback.called


class TestRaptorIndexProcessor:
    """Tests for RaptorIndexProcessor class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        llm.embed_query.return_value = [0.1, 0.2, 0.3]
        llm.chat.return_value = "Summary text"
        return llm

    @pytest.fixture
    def mock_vector_manager(self):
        """Create a mock vector manager."""
        return MagicMock()

    @pytest.fixture
    def mock_splitter(self):
        """Create a mock splitter."""
        splitter = MagicMock()
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

    def test_init_with_all_components(self, mock_llm, mock_vector_manager, mock_splitter):
        """Initialize with all components."""
        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
            splitter=mock_splitter,
        )

        assert processor.llm == mock_llm
        assert processor.vector_manager == mock_vector_manager
        assert processor.splitter == mock_splitter
        assert processor.cleaner is not None
        assert processor.config is not None

    def test_init_with_custom_config(self, mock_llm):
        """Initialize with custom config."""
        config = RaptorConfig(max_cluster=32)
        processor = RaptorIndexProcessor(llm=mock_llm, config=config)

        assert processor.config.max_cluster == 32

    def test_init_with_separate_embedder(self, mock_llm):
        """Initialize with separate embedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]

        processor = RaptorIndexProcessor(llm=mock_llm, embedder=embedder)

        assert processor.embedder == embedder

    def test_embed_texts_uses_embedder(self, mock_llm):
        """Uses embedder when provided."""
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]

        processor = RaptorIndexProcessor(llm=mock_llm, embedder=embedder)
        result = processor._embed_texts(["test"])

        embedder.embed.assert_called_once_with(["test"])
        assert result == [[0.1, 0.2]]

    def test_embed_texts_uses_llm(self, mock_llm):
        """Uses LLM when embedder not provided."""
        processor = RaptorIndexProcessor(llm=mock_llm)
        processor._embed_texts(["test"])

        mock_llm.embed_documents.assert_called_once_with(["test"])

    def test_process_empty_documents(self, mock_llm, mock_vector_manager, sample_dataset):
        """Process empty document list."""
        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, [])

        mock_llm.embed_documents.assert_not_called()
        mock_vector_manager.add_texts.assert_not_called()

    def test_process_adds_dataset_id(
        self, mock_llm, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Dataset ID is added to chunk metadata."""
        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, sample_documents)

        call_args = mock_vector_manager.add_texts.call_args
        chunks = call_args[0][1]
        for chunk in chunks:
            assert chunk.metadata["dataset_id"] == "test-dataset"

    def test_process_adds_raptor_layer(
        self, mock_llm, mock_vector_manager, sample_dataset, sample_documents
    ):
        """RAPTOR layer is added to chunk metadata."""
        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, sample_documents)

        call_args = mock_vector_manager.add_texts.call_args
        chunks = call_args[0][1]
        for chunk in chunks:
            assert "raptor_layer" in chunk.metadata

    def test_process_with_callback(
        self, mock_llm, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Progress callback is called."""
        callback = MagicMock()
        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
            progress_callback=callback,
        )

        processor.process(sample_dataset, sample_documents)

        assert callback.called

    @patch("langrag.datasource.vdb.global_manager.get_vector_manager")
    def test_process_uses_global_manager(
        self, mock_get_manager, mock_llm, sample_dataset, sample_documents
    ):
        """Uses global vector manager when not provided."""
        mock_global_manager = MagicMock()
        mock_get_manager.return_value = mock_global_manager

        processor = RaptorIndexProcessor(llm=mock_llm, vector_manager=None)
        processor.process(sample_dataset, sample_documents)

        mock_get_manager.assert_called_once()
        mock_global_manager.add_texts.assert_called_once()

    def test_process_handles_raptor_failure(
        self, mock_llm, mock_vector_manager, sample_dataset, sample_documents
    ):
        """Falls back to standard indexing on RAPTOR failure."""
        mock_llm.chat.side_effect = Exception("LLM error")

        processor = RaptorIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
            config=RaptorConfig(max_errors=1),
        )

        # Should not raise, falls back to standard indexing
        processor.process(sample_dataset, sample_documents)

        mock_vector_manager.add_texts.assert_called_once()
