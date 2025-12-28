"""纯全文检索 Provider"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..base import BaseRetrievalProvider

if TYPE_CHECKING:
    from langrag.core.search_result import SearchResult
    from langrag.vector_store import BaseVectorStore


class FullTextSearchProvider(BaseRetrievalProvider):
    """纯全文关键词检索

    使用 BM25 等算法进行关键词匹配。
    适用于支持全文检索的存储（如 DuckDB、SeekDB）。
    """

    def __init__(self, vector_store: BaseVectorStore, name: str = None):
        """初始化全文检索 Provider

        Args:
            vector_store: 向量存储（需支持全文检索）
            name: Provider 名称
        """
        super().__init__(name)
        self.vector_store = vector_store

        # 验证存储能力
        if not vector_store.capabilities.supports_fulltext:
            raise ValueError(
                f"Vector store {vector_store.__class__.__name__} does not support full-text search"
            )

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """执行全文检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        logger.debug(f"[{self.name}] Full-text search for: {query[:50]}...")

        # 全文检索
        results = self.vector_store.search_fulltext(query_text=query, top_k=top_k)

        logger.debug(f"[{self.name}] Retrieved {len(results)} results")
        return results
