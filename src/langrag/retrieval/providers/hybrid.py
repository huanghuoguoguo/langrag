"""混合检索 Provider"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from langrag.utils.rrf import reciprocal_rank_fusion

from ..base import BaseRetrievalProvider

if TYPE_CHECKING:
    from langrag.core.search_result import SearchResult
    from langrag.embedder import BaseEmbedder
    from langrag.vector_store import BaseVectorStore


class HybridSearchProvider(BaseRetrievalProvider):
    """混合检索（向量 + 全文）

    支持两种模式：
    1. 原生混合检索：VDB 自身支持（如 SeekDB）
    2. 应用层融合：分别执行向量和全文检索，然后用 RRF 融合

    自动根据 VDB 能力选择最佳模式。
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        alpha: float = 0.5,
        rrf_k: int = 60,
        name: str = None,
    ):
        """初始化混合检索 Provider

        Args:
            embedder: 嵌入器
            vector_store: 向量存储
            alpha: 向量/全文权重（用于原生混合检索，0=纯全文, 1=纯向量）
            rrf_k: RRF 常数（用于应用层融合）
            name: Provider 名称
        """
        super().__init__(name)
        self.embedder = embedder
        self.vector_store = vector_store
        self.alpha = alpha
        self.rrf_k = rrf_k

        caps = vector_store.capabilities

        # 检查能力并确定模式
        if caps.supports_hybrid:
            # 使用原生混合检索
            self.mode = "native"
            logger.info(f"[{self.name}] Using native hybrid search")
        elif caps.supports_vector and caps.supports_fulltext:
            # 使用应用层 RRF 融合
            self.mode = "rrf_fusion"
            logger.info(f"[{self.name}] Using RRF fusion (vector + fulltext)")
        else:
            raise ValueError(
                f"Vector store {vector_store.__class__.__name__} "
                f"does not support hybrid search (requires both vector and fulltext)"
            )

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """执行混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        logger.debug(f"[{self.name}] Hybrid search for: {query[:50]}...")

        if self.mode == "native":
            # 使用 VDB 原生混合检索
            return await self._native_hybrid_search(query, top_k)
        else:
            # 使用应用层 RRF 融合
            return await self._rrf_fusion_search(query, top_k)

    async def _native_hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        """原生混合检索（VDB 内部实现）"""
        query_embedding = self.embedder.embed([query])[0]

        results = self.vector_store.search_hybrid(
            query_vector=query_embedding, query_text=query, alpha=self.alpha, top_k=top_k
        )

        logger.debug(f"[{self.name}] Native hybrid retrieved {len(results)} results")
        return results

    async def _rrf_fusion_search(self, query: str, top_k: int) -> list[SearchResult]:
        """应用层 RRF 融合（向量 + 全文分别检索后融合）"""
        # 每个检索多取一些候选，便于融合
        candidate_k = max(top_k * 2, 20)

        # 1. 向量检索
        query_embedding = self.embedder.embed([query])[0]
        vector_results = self.vector_store.search(query_vector=query_embedding, top_k=candidate_k)
        logger.debug(f"[{self.name}] Vector: {len(vector_results)} results")

        # 2. 全文检索
        fulltext_results = self.vector_store.search_fulltext(query_text=query, top_k=candidate_k)
        logger.debug(f"[{self.name}] Fulltext: {len(fulltext_results)} results")

        # 3. RRF 融合
        fused_results = reciprocal_rank_fusion(
            [vector_results, fulltext_results], k=self.rrf_k, top_k=top_k
        )

        logger.debug(f"[{self.name}] RRF fused {len(fused_results)} results")
        return fused_results
