"""纯向量检索 Provider"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..base import BaseRetrievalProvider

if TYPE_CHECKING:
    from langrag.core.search_result import SearchResult
    from langrag.embedder import BaseEmbedder
    from langrag.vector_store import BaseVectorStore


class VectorSearchProvider(BaseRetrievalProvider):
    """纯向量相似度检索

    使用向量嵌入进行语义相似度搜索。
    适用于所有支持向量检索的存储。
    """

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore, name: str = None):
        """初始化向量检索 Provider

        Args:
            embedder: 嵌入器（用于查询向量化）
            vector_store: 向量存储
            name: Provider 名称
        """
        super().__init__(name)
        self.embedder = embedder
        self.vector_store = vector_store

        # 验证存储能力
        if not vector_store.capabilities.supports_vector:
            raise ValueError(
                f"Vector store {vector_store.__class__.__name__} does not support vector search"
            )

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """执行向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        logger.debug(f"[{self.name}] Vector search for: {query[:50]}...")

        # 1. 向量化查询
        query_embedding = self.embedder.embed([query])[0]

        # 2. 向量检索
        results = self.vector_store.search(query_vector=query_embedding, top_k=top_k)

        logger.debug(f"[{self.name}] Retrieved {len(results)} results")
        return results
