from abc import ABC, abstractmethod
from typing import List, Any
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

class VectorManager(ABC):
    """
    RAG 系统的统一向量/存储管理器接口。
    所有的外部调用（Search, Add, Delete）都应该通过这个管理器进行，而不再是直接操作 VDB 实例。
    
    这个管理器负责路由：
    1. 根据 dataset 找到对应的 VDB 实现。
    2. 或者直接转发给外部定义的处理逻辑（Bridge Mode）。
    """
    
    @abstractmethod
    def search(
        self, 
        dataset: Dataset, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> List[Document]:
        """统一的搜索入口"""
        pass

    @abstractmethod
    def add_texts(
        self, 
        dataset: Dataset, 
        documents: List[Document], 
        **kwargs
    ) -> None:
        """统一的数据添加入口"""
        pass
        
    @abstractmethod
    def delete(self, dataset: Dataset) -> None:
        pass


class DefaultVectorManager(VectorManager):
    """
    默认的管理器实现。
    它继续使用之前的 VDB Factory 逻辑来实例化具体的 VDB 类，并调用之。
    """
    def search(self, dataset: Dataset, query: str, query_vector: list[float] | None, top_k: int = 4, **kwargs) -> List[Document]:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        # 通过 Factory 获取具体的 VDB 实例 (可能是 DuckDB, Chroma, 或 BridgeVector)
        store = VectorStoreFactory.get_vector_store(dataset)
        return store.search(query, query_vector, top_k=top_k, **kwargs)

    def add_texts(self, dataset: Dataset, documents: List[Document], **kwargs) -> None:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        store = VectorStoreFactory.get_vector_store(dataset)
        store.add_texts(documents, **kwargs)
        
    def delete(self, dataset: Dataset) -> None:
        from langrag.datasource.vdb.factory import VectorStoreFactory
        store = VectorStoreFactory.get_vector_store(dataset)
        store.delete()
