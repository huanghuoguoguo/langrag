"""Base class for context compressors."""

from abc import ABC, abstractmethod

from ..core.search_result import SearchResult


class BaseCompressor(ABC):
    """Abstract base class for context compressors.

    上下文压缩器的基类，用于在传给 LLM 之前压缩检索结果，减少 token 数量。

    典型用途：
    - 提取关键句子
    - 过滤冗余内容
    - 总结长文本
    """

    @abstractmethod
    def compress(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """压缩检索结果的上下文内容

        Args:
            query: 用户查询
            results: 检索结果列表
            target_ratio: 目标压缩比率（0-1），例如 0.5 表示压缩到 50% 长度

        Returns:
            压缩后的检索结果列表，每个结果的 content 可能被压缩
        """
        pass
