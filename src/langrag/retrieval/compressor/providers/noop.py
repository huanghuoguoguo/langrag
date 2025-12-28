"""No-op compressor that returns results unchanged."""

from loguru import logger

from ...core.search_result import SearchResult
from ..base import BaseCompressor


class NoOpCompressor(BaseCompressor):
    """Pass-through compressor that performs no compression.
    
    不进行压缩的占位符实现，直接返回原始结果。
    """

    def compress(
        self,
        query: str,  # noqa: ARG002
        results: list[SearchResult],
        target_ratio: float = 0.5,  # noqa: ARG002
    ) -> list[SearchResult]:
        """Return results unchanged.
        
        Args:
            query: 用户查询（未使用）
            results: 检索结果列表
            target_ratio: 目标压缩比率（未使用）
            
        Returns:
            原始结果，未经压缩
        """
        logger.debug("NoOpCompressor: returning original results without compression")
        return results

