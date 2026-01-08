"""Tests for WebVectorStoreManager."""

from unittest.mock import MagicMock, patch

import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class TestWebVectorStoreManager:
    """Tests for WebVectorStoreManager class."""

    def _create_dataset(self, vdb_type: str = "chroma") -> Dataset:
        """Helper to create a Dataset."""
        return Dataset(
            id="test-kb-id",
            name="Test KB",
            collection_name="test_collection",
            vdb_type=vdb_type,
        )

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("web.core.vdb_manager.DUCKDB_DIR", "/tmp/duckdb")
    @patch("web.core.vdb_manager.SEEKDB_DIR", "/tmp/seekdb")
    def test_init(self):
        """Manager initializes with empty stores."""
        from web.core.vdb_manager import WebVectorStoreManager

        manager = WebVectorStoreManager()
        assert manager._stores == {}

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("langrag.datasource.vdb.chroma.ChromaVector")
    def test_create_store_chroma(self, mock_chroma_class):
        """Create ChromaDB store."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset(vdb_type="chroma")

        store = manager.create_store(dataset)

        assert store == mock_store
        mock_chroma_class.assert_called_once()
        assert dataset.id in manager._stores

    @patch("web.core.vdb_manager.DUCKDB_DIR", "/tmp/duckdb")
    @patch("langrag.datasource.vdb.duckdb.DuckDBVector")
    def test_create_store_duckdb(self, mock_duckdb_class):
        """Create DuckDB store."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_duckdb_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset(vdb_type="duckdb")

        store = manager.create_store(dataset)

        assert store == mock_store
        mock_duckdb_class.assert_called_once()

    @patch("web.core.vdb_manager.SEEKDB_DIR", "/tmp/seekdb")
    @patch("langrag.datasource.vdb.seekdb.SeekDBVector")
    def test_create_store_seekdb(self, mock_seekdb_class):
        """Create SeekDB store."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_seekdb_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset(vdb_type="seekdb")

        store = manager.create_store(dataset)

        assert store == mock_store
        mock_seekdb_class.assert_called_once()

    def test_create_store_unsupported_type(self):
        """Unsupported VDB type raises error."""
        from web.core.vdb_manager import WebVectorStoreManager

        manager = WebVectorStoreManager()
        dataset = self._create_dataset(vdb_type="unknown")

        with pytest.raises(ValueError, match="Unsupported vector database type"):
            manager.create_store(dataset)

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("langrag.datasource.vdb.chroma.ChromaVector")
    def test_get_vector_store_caches(self, mock_chroma_class):
        """get_vector_store caches and reuses stores."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset()

        # First call creates store
        store1 = manager.get_vector_store(dataset)
        # Second call returns cached store
        store2 = manager.get_vector_store(dataset)

        assert store1 == store2
        # ChromaVector should only be created once
        mock_chroma_class.assert_called_once()

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("langrag.datasource.vdb.chroma.ChromaVector")
    def test_search(self, mock_chroma_class):
        """search delegates to underlying store."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_store.search.return_value = [
            Document(page_content="result", metadata={})
        ]
        mock_chroma_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset()

        results = manager.search(dataset, "query", [0.1, 0.2], top_k=5)

        assert len(results) == 1
        mock_store.search.assert_called_once_with(
            "query", [0.1, 0.2], top_k=5
        )

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("langrag.datasource.vdb.chroma.ChromaVector")
    def test_add_texts(self, mock_chroma_class):
        """add_texts delegates to underlying store."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset()
        docs = [Document(page_content="doc1", metadata={})]

        manager.add_texts(dataset, docs)

        mock_store.add_texts.assert_called_once_with(docs)

    @patch("web.core.vdb_manager.CHROMA_DIR", "/tmp/chroma")
    @patch("langrag.datasource.vdb.chroma.ChromaVector")
    def test_delete(self, mock_chroma_class):
        """delete removes store from cache."""
        from web.core.vdb_manager import WebVectorStoreManager

        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store

        manager = WebVectorStoreManager()
        dataset = self._create_dataset()

        # First get the store to cache it
        manager.get_vector_store(dataset)
        assert dataset.id in manager._stores

        # Delete should remove from cache
        manager.delete(dataset)
        mock_store.delete.assert_called_once()
        assert dataset.id not in manager._stores

    def test_default_vdb_type_is_seekdb(self):
        """When vdb_type is None, defaults to seekdb."""
        from web.core.vdb_manager import WebVectorStoreManager

        dataset = Dataset(
            id="test-kb",
            name="Test",
            collection_name="test_coll",
            vdb_type=None,
        )

        manager = WebVectorStoreManager()

        with patch("web.core.vdb_manager.SEEKDB_DIR", "/tmp/seekdb"):
            with patch("langrag.datasource.vdb.seekdb.SeekDBVector") as mock_seekdb:
                mock_seekdb.return_value = MagicMock()
                manager.create_store(dataset)
                mock_seekdb.assert_called_once()
