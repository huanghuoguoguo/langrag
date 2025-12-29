from typing import Any, Type
from loguru import logger

from langrag.entities.dataset import Dataset
from langrag.datasource.vdb.base import BaseVector
from langrag.datasource.vdb.chroma import ChromaVector
from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.datasource.vdb.seekdb import SeekDBVector
from langrag.config.settings import settings


class VectorStoreFactory:
    """
    Factory for creating Vector Store instances based on type.
    Now used primarily by the DefaultVectorManager.
    """

    _registry: dict[str, Type[BaseVector]] = {
        "chroma": ChromaVector,
        "duckdb": DuckDBVector,
        "seekdb": SeekDBVector,
    }

    @classmethod
    def create(cls, type_name: str, dataset: Dataset, **params: Any) -> BaseVector:
        """
        Create a vector store instance.
        
        Args:
            type_name: Type identifier (e.g., "chroma", "duckdb")
            dataset: The dataset to manage
            **params: Additional configuration parameters
        """
        if not type_name:
            type_name = "seekdb" # Default fallback
            
        if type_name not in cls._registry:
             available = ", ".join(cls._registry.keys())
             raise ValueError(f"Unknown Vector Store Type: '{type_name}'. Available types: {available}")

        vdb_class = cls._registry[type_name]
        logger.debug(f"Creating {vdb_class.__name__} for dataset {dataset.name} with params: {params}")

        # Inject default paths if not provided and using local defaults
        if type_name == "chroma" and "persist_directory" not in params and "host" not in params:
            params["persist_directory"] = settings.CHROMA_DB_PATH
        
        if type_name == "duckdb" and "database_path" not in params:
            params["database_path"] = settings.DUCKDB_PATH

        return vdb_class(dataset, **params)

    @classmethod
    def get_vector_store(cls, dataset: Dataset, type_name: str | None = None, **kwargs) -> BaseVector:
        """
        Helper to create VDB from dataset config.
        """
        if not type_name:
             type_name = dataset.vdb_type
        
        if not type_name:
            type_name = "seekdb"

        # Handle special case for web_search which is not in registry
        if type_name == "web_search":
            from langrag.datasource.vdb.web import WebVector
            return WebVector(dataset)
        
        # Separate specific kwargs if needed, but for now passing all
        return cls.create(type_name, dataset, **kwargs)
