import pytest
from unittest.mock import MagicMock, patch
from langrag.entities.dataset import Dataset
from langrag.datasource.vdb.factory import VectorStoreFactory
from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.datasource.vdb.chroma import ChromaVector

class TestVectorStoreFactory:

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_duckdb_default(self, mock_settings):
        mock_settings.DUCKDB_PATH = "/tmp/duck"
        
        dataset = Dataset(name="ds", collection_name="col")
        # Should default to duckdb
        with patch("langrag.datasource.vdb.factory.DuckDBVector") as mock_cls:
            store = VectorStoreFactory.get_vector_store(dataset)
            mock_cls.assert_called_with(dataset, database_path="/tmp/duck")

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_chroma_explicit(self, mock_settings):
        mock_settings.CHROMA_DB_PATH = "/tmp/chroma"
        
        dataset = Dataset(name="ds", collection_name="col")
        with patch("langrag.datasource.vdb.factory.ChromaVector") as mock_cls:
            store = VectorStoreFactory.get_vector_store(dataset, type_name="chroma")
            mock_cls.assert_called_with(dataset, persist_directory="/tmp/chroma")

    @patch("langrag.datasource.vdb.factory.settings")
    def test_get_vector_store_from_dataset_config(self, mock_settings):
        # Assuming vdb_type field exists
        dataset = Dataset(name="ds", collection_name="col", vdb_type="seekdb")
        
        with patch("langrag.datasource.vdb.factory.SeekDBVector") as mock_cls:
            store = VectorStoreFactory.get_vector_store(dataset)
            mock_cls.assert_called_with(dataset)
