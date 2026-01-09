import logging

from langrag.datasource.vdb.base import BaseVector
from langrag.datasource.vdb.vector_manager import VectorManager
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from web.config import CHROMA_DIR, DUCKDB_DIR, SEEKDB_DIR

logger = logging.getLogger(__name__)

class WebVectorStoreManager(VectorManager):
    """
    Web 层的向量数据库管理器。
    负责管理 VDB 实例的生命周期，根据 Web 配置创建实例，并缓存实例。
    实现了 LangRAG 的 BaseVectorStoreManager 接口，可注入核心层。
    """

    def __init__(self):
        self._stores: dict[str, BaseVector] = {}

    def get_vector_store(self, dataset: Dataset, **kwargs) -> BaseVector:
        """
        获取或创建向量存储实例。
        优先查找缓存。
        """
        kb_id = dataset.id
        if kb_id in self._stores:
            return self._stores[kb_id]

        return self.create_store(dataset)

    def create_store(self, dataset: Dataset) -> BaseVector:
        """根据数据集配置创建 VDB 实例"""

        # Local Mode Only
        vdb_type = dataset.vdb_type or "seekdb" # 默认 SeekDB

        logger.info(f"[WebManager] Creating/Loading Local store for KB: {dataset.id}, type: {vdb_type}")

        store = None
        if vdb_type == "chroma":
            from langrag.datasource.vdb.chroma import ChromaVector
            store = ChromaVector(dataset, persist_directory=str(CHROMA_DIR))

        elif vdb_type == "duckdb":
            from langrag.datasource.vdb.duckdb import DuckDBVector
            store = DuckDBVector(dataset, database_path=str(DUCKDB_DIR))

        elif vdb_type == "seekdb":
            from langrag.datasource.vdb.seekdb import SeekDBVector
            store = SeekDBVector(dataset, mode="embedded", db_path=str(SEEKDB_DIR))

        else:
            raise ValueError(f"Unsupported vector database type: {vdb_type}")

        self._stores[dataset.id] = store
        return store

    def search(
        self,
        dataset: Dataset,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        store = self.get_vector_store(dataset)
        return store.search(query, query_vector, top_k=top_k, **kwargs)

    def add_texts(
        self,
        dataset: Dataset,
        documents: list[Document],
        **kwargs
    ) -> None:
        store = self.get_vector_store(dataset)
        store.add_texts(documents, **kwargs)

    def delete(self, dataset: Dataset) -> None:
        store = self.get_vector_store(dataset)
        store.delete()
        if dataset.id in self._stores:
            del self._stores[dataset.id]
