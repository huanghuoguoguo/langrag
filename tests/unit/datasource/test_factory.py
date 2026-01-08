import pytest
from unittest.mock import MagicMock, patch
from langrag.entities.dataset import Dataset
from langrag.datasource.vdb.factory import VectorStoreFactory

class TestVectorStoreFactory:

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_seekdb_default(self, mock_settings):
        # Default is now SeekDB
        dataset = Dataset(name="ds", collection_name="col")
        
        mock_cls = MagicMock()
        mock_cls.__name__ = "MockSeekDB"
        with patch.dict(VectorStoreFactory._registry, {"seekdb": mock_cls}):
            store = VectorStoreFactory.get_vector_store(dataset)
            mock_cls.assert_called_with(dataset)

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_duckdb_explicit(self, mock_settings):
        mock_settings.DUCKDB_PATH = "/tmp/duck"
        
        dataset = Dataset(name="ds", collection_name="col")
        
        mock_cls = MagicMock()
        mock_cls.__name__ = "MockDuckDB"
        with patch.dict(VectorStoreFactory._registry, {"duckdb": mock_cls}):
            store = VectorStoreFactory.get_vector_store(dataset, type_name="duckdb")
            mock_cls.assert_called_with(dataset, database_path="/tmp/duck")

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_chroma_explicit(self, mock_settings):
        mock_settings.CHROMA_DB_PATH = "/tmp/chroma"
        
        dataset = Dataset(name="ds", collection_name="col")
        
        mock_cls = MagicMock()
        mock_cls.__name__ = "MockChroma"
        with patch.dict(VectorStoreFactory._registry, {"chroma": mock_cls}):
            store = VectorStoreFactory.get_vector_store(dataset, type_name="chroma")
            mock_cls.assert_called_with(dataset, persist_directory="/tmp/chroma")

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_from_dataset_config_duckdb(self, mock_settings):
        mock_settings.DUCKDB_PATH = "/tmp/duck"
        dataset = Dataset(name="ds", collection_name="col", vdb_type="duckdb")
        
        mock_cls = MagicMock()
        mock_cls.__name__ = "MockDuckDB"
        with patch.dict(VectorStoreFactory._registry, {"duckdb": mock_cls}):
            store = VectorStoreFactory.get_vector_store(dataset)
            mock_cls.assert_called_with(dataset, database_path="/tmp/duck")
