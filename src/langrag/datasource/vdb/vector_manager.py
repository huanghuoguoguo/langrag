from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class VectorManager(ABC):
    """
    Unified vector/storage manager interface for RAG systems.
    All external calls (Search, Add, Delete) should go through this manager, rather than directly operating VDB instances.

    This manager is responsible for routing:
    1. Finding the corresponding VDB implementation based on dataset.
    2. Or forwarding directly to externally defined processing logic (Bridge Mode).
    """

    @abstractmethod
    def search(
        self,
        dataset: Dataset,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Unified search entry point"""
        pass

    @abstractmethod
    def add_texts(
        self,
        dataset: Dataset,
        documents: list[Document],
        **kwargs
    ) -> None:
        """Unified data addition entry point"""
        pass

    @abstractmethod
    def delete(self, dataset: Dataset) -> None:
        pass


class DefaultVectorManager(VectorManager):
    """
    Default manager implementation.
    It continues to use the previous VDB Factory logic to instantiate specific VDB classes and call them.
    """
    def search(self, dataset: Dataset, query: str, query_vector: list[float] | None, top_k: int = 4, **kwargs) -> list[Document]:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        # Get specific VDB instance via Factory (could be DuckDB, Chroma, or BridgeVector)
        store = VectorStoreFactory.get_vector_store(dataset)
        return store.search(query, query_vector, top_k=top_k, **kwargs)

    def add_texts(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        store = VectorStoreFactory.get_vector_store(dataset)
        store.add_texts(documents, **kwargs)

    def delete(self, dataset: Dataset) -> None:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        store = VectorStoreFactory.get_vector_store(dataset)
        store.delete()
