from typing import Any
from langrag.entities.dataset import Dataset
from langrag.datasource.vdb.base import BaseVector
from langrag.datasource.vdb.chroma import ChromaVector
from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.datasource.vdb.seekdb import SeekDBVector
from langrag.config.settings import settings

class VectorStoreFactory:
    """Factory for creating Vector Store instances."""

    @staticmethod
    def get_vector_store(dataset: Dataset, type_name: str | None = None) -> BaseVector:
        """
        Get vector store instance.
        Priority:
        1. Explicit type_name argument
        2. Dataset vdb_type field
        3. Default (DuckDB)
        """
        if not type_name:
             type_name = dataset.vdb_type
        
        if not type_name:
            type_name = "duckdb"

        if type_name == "chroma":
            return ChromaVector(dataset, persist_directory=settings.CHROMA_DB_PATH)
        elif type_name == "duckdb":
            return DuckDBVector(dataset, database_path=settings.DUCKDB_PATH)
        elif type_name == "seekdb":
            return SeekDBVector(dataset)
        else:
            raise ValueError(f"Unknown Vector Store Type: {type_name}")
