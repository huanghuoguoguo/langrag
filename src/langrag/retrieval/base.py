"""基础检索提供者接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.search_result import SearchResult


class BaseRetrievalProvider(ABC):
    """检索提供者基类

    每个 Provider 代表一种检索策略，如：
    - VectorSearchProvider: 纯向量检索
    - FullTextSearchProvider: 纯全文检索
    - HybridSearchProvider: 混合检索（向量+全文）

    Provider 从 VectorStore 获取数据，但屏蔽底层存储细节。
    """

    def __init__(self, name: str = None):
        """初始化 Provider

        Args:
            name: Provider 名称（用于日志和调试）
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """执行检索

        Args:
            query: 查询文本
            top_k: 返回的结果数量

        Returns:
            检索结果列表
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
