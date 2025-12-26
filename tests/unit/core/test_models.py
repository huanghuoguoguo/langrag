"""Unit tests for core data models.

单元测试特点：
- 测试数据模型的创建和验证
- 测试字段验证逻辑
- 快速执行
"""

import pytest
from langrag import Document, Chunk, SearchResult


@pytest.mark.unit
class TestDocument:
    """Test Document model."""

    def test_create_document(self):
        """Document can be created with required fields."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt"}
        )

        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.id is not None  # Auto-generated ID

    def test_document_with_custom_id(self):
        """Document can be created with custom ID."""
        doc = Document(
            id="custom-123",
            content="Test content",
            metadata={}
        )

        assert doc.id == "custom-123"

    def test_document_empty_content(self):
        """Document validates non-empty content."""
        with pytest.raises(ValueError):
            Document(content="", metadata={})


@pytest.mark.unit
class TestChunk:
    """Test Chunk model."""

    def test_create_chunk(self):
        """Chunk can be created with required fields."""
        chunk = Chunk(
            content="Chunk content",
            embedding=[0.1, 0.2, 0.3],
            source_doc_id="doc-123",
            metadata={"page": 1}
        )

        assert chunk.content == "Chunk content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.source_doc_id == "doc-123"
        assert chunk.metadata["page"] == 1
        assert chunk.id is not None  # Auto-generated

    def test_chunk_empty_content(self):
        """Chunk validates non-empty content."""
        with pytest.raises(ValueError):
            Chunk(
                content="",
                embedding=[0.1],
                source_doc_id="doc-1",
                metadata={}
            )


@pytest.mark.unit
class TestSearchResult:
    """Test SearchResult model."""

    def test_create_search_result(self):
        """SearchResult can be created with chunk and score."""
        chunk = Chunk(
            content="Result content",
            embedding=[0.5],
            source_doc_id="doc-1",
            metadata={}
        )

        result = SearchResult(chunk=chunk, score=0.95)

        assert result.chunk == chunk
        assert result.score == 0.95

    def test_search_result_score_validation(self):
        """SearchResult validates score is between 0 and 1."""
        chunk = Chunk(
            content="Content",
            embedding=[0.1],
            source_doc_id="doc-1",
            metadata={}
        )

        # Valid scores
        SearchResult(chunk=chunk, score=0.0)
        SearchResult(chunk=chunk, score=0.5)
        SearchResult(chunk=chunk, score=1.0)

        # Invalid scores should raise
        with pytest.raises(ValueError):
            SearchResult(chunk=chunk, score=-0.1)

        with pytest.raises(ValueError):
            SearchResult(chunk=chunk, score=1.5)
