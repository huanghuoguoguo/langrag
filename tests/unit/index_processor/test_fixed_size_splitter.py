"""Tests for FixedSizeChunker."""

import pytest

from langrag.entities.document import Document, DocumentType
from langrag.index_processor.splitter.providers.fixed_size import FixedSizeChunker


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker class."""

    def test_init_default_values(self):
        """Default initialization values."""
        chunker = FixedSizeChunker()
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    def test_init_custom_values(self):
        """Custom initialization values."""
        chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100

    def test_init_invalid_chunk_size(self):
        """Raises error for invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=-1)

    def test_init_invalid_overlap(self):
        """Raises error for invalid overlap."""
        with pytest.raises(ValueError, match="overlap must be in"):
            FixedSizeChunker(chunk_size=100, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be in"):
            FixedSizeChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError, match="overlap must be in"):
            FixedSizeChunker(chunk_size=100, overlap=150)

    def test_split_single_document(self):
        """Splits a single document into chunks."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        doc = Document(page_content="0123456789ABCDEFGHIJ", metadata={"source": "test"})

        chunks = chunker.split([doc])

        assert len(chunks) > 1
        # Each chunk should have metadata
        for chunk in chunks:
            assert "source_doc_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "start_char" in chunk.metadata
            assert "end_char" in chunk.metadata
            assert chunk.metadata.get("type") == DocumentType.CHUNK

    def test_split_preserves_metadata(self):
        """Original metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        doc = Document(
            page_content="0123456789ABCDEFGHIJ",
            metadata={"source": "test", "custom_field": "value"},
        )

        chunks = chunker.split([doc])

        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["custom_field"] == "value"

    def test_split_empty_document(self):
        """Handles very short document that produces single chunk."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        doc = Document(page_content="X", metadata={})  # Minimum content

        chunks = chunker.split([doc])

        assert len(chunks) == 1

    def test_split_document_smaller_than_chunk_size(self):
        """Document smaller than chunk_size produces single chunk."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = Document(page_content="Small text", metadata={})

        chunks = chunker.split([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == "Small text"

    def test_split_multiple_documents(self):
        """Splits multiple documents."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        docs = [
            Document(page_content="0123456789", metadata={"doc": 1}),
            Document(page_content="ABCDEFGHIJ", metadata={"doc": 2}),
        ]

        chunks = chunker.split(docs)

        assert len(chunks) == 2
        assert chunks[0].metadata["doc"] == 1
        assert chunks[1].metadata["doc"] == 2

    def test_split_overlap_functionality(self):
        """Overlap creates overlapping content between chunks."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        doc = Document(page_content="0123456789ABCDEFGHIJ", metadata={})

        chunks = chunker.split([doc])

        # With overlap=3 and chunk_size=10, second chunk starts at position 7
        assert len(chunks) >= 2
        # Check overlap: end of first chunk overlaps with start of second
        if len(chunks) >= 2:
            # First chunk: 0-9 (0123456789)
            # Second chunk starts at 7: 789ABCDEFG
            assert chunks[0].page_content == "0123456789"
            assert chunks[1].page_content.startswith("789")

    def test_chunk_metadata_positions(self):
        """Chunk metadata contains correct position info."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        doc = Document(page_content="0123456789ABCDEFGHIJ", metadata={})

        chunks = chunker.split([doc])

        assert chunks[0].metadata["start_char"] == 0
        assert chunks[0].metadata["end_char"] == 10
        assert chunks[0].metadata["chunk_index"] == 0

        assert chunks[1].metadata["start_char"] == 10
        assert chunks[1].metadata["end_char"] == 20
        assert chunks[1].metadata["chunk_index"] == 1

    def test_split_no_overlap(self):
        """No overlap when overlap=0."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        doc = Document(page_content="0123456789ABCDEFGHIJ", metadata={})

        chunks = chunker.split([doc])

        assert chunks[0].page_content == "0123456789"
        assert chunks[1].page_content == "ABCDEFGHIJ"
