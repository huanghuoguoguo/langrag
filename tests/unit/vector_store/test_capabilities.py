"""Unit tests for VectorStore capabilities system.

单元测试特点：
- 测试能力声明机制
- 验证能力验证逻辑
- 测试不支持的操作
"""

import pytest

from langrag import Chunk, InMemoryVectorStore, SearchMode, VectorStoreCapabilities


@pytest.mark.unit
class TestVectorStoreCapabilities:
    """Test capability detection for vector stores."""

    def test_in_memory_capabilities(self):
        """InMemoryVectorStore should only support vector search."""
        store = InMemoryVectorStore()
        caps = store.capabilities

        assert caps.supports_vector is True
        assert caps.supports_fulltext is False
        assert caps.supports_hybrid is False

    def test_capability_validation(self):
        """Capability validation should raise for unsupported modes."""
        caps = VectorStoreCapabilities(
            supports_vector=True, supports_fulltext=False, supports_hybrid=False
        )

        # Should not raise for supported mode
        caps.validate_mode(SearchMode.VECTOR)

        # Should raise for unsupported modes
        with pytest.raises(ValueError, match="Full-text search not supported"):
            caps.validate_mode(SearchMode.FULLTEXT)

        with pytest.raises(ValueError, match="Hybrid search not supported"):
            caps.validate_mode(SearchMode.HYBRID)

    def test_unsupported_search_methods(self):
        """InMemoryVectorStore should raise for unsupported search methods."""
        store = InMemoryVectorStore()

        # Add some test data
        chunks = [
            Chunk(
                content="Test content", embedding=[0.1, 0.2, 0.3], source_doc_id="doc1", metadata={}
            )
        ]
        store.add(chunks)

        # Full-text search should raise
        with pytest.raises(NotImplementedError, match="does not support full-text search"):
            store.search_fulltext("test query")

        # Hybrid search should raise
        with pytest.raises(NotImplementedError, match="does not support native hybrid search"):
            store.search_hybrid(query_vector=[0.1, 0.2, 0.3], query_text="test query")
