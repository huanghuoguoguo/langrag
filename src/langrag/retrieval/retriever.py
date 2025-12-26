"""检索协调器 - 管理多个检索 Provider 并融合结果"""

from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING
from loguru import logger

from .base import BaseRetrievalProvider
from .factory import ProviderFactory
from ..utils.rrf import reciprocal_rank_fusion, weighted_rrf

if TYPE_CHECKING:
    from ..embedder import BaseEmbedder
    from ..vector_store import BaseVectorStore
    from ..core.search_result import SearchResult
    from ..config.models import StorageRole


class Retriever:
    """检索协调器
    
    管理多个检索 Provider，支持：
    1. 单源检索（能力自适应）
    2. 多源检索（角色分工 + RRF 融合）
    
    智能特性：
    - 自动根据 VDB 能力选择最佳检索策略
    - 支持多数据源并行检索和结果融合
    - 灵活的 Provider 配置和权重调整
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        providers: list[BaseRetrievalProvider] | None = None,
        fusion_strategy: str = "rrf",
        fusion_weights: list[float] | None = None,
        rrf_k: int = 60
    ):
        """初始化检索协调器
        
        Args:
            embedder: 嵌入器（用于向量化查询）
            providers: 检索 Provider 列表（如果为空则需后续添加）
            fusion_strategy: 融合策略 ("rrf" | "weighted_rrf")
            fusion_weights: Provider 权重（用于 weighted_rrf）
            rrf_k: RRF 常数
        """
        self.embedder = embedder
        self.providers = providers or []
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights
        self.rrf_k = rrf_k
        
        # 验证权重
        if fusion_weights and len(fusion_weights) != len(self.providers):
            raise ValueError(
                f"Fusion weights length ({len(fusion_weights)}) "
                f"must match providers length ({len(self.providers)})"
            )
        
        logger.info(
            f"Initialized Retriever with {len(self.providers)} providers: "
            f"{[p.name for p in self.providers]}"
        )

    def add_provider(self, provider: BaseRetrievalProvider) -> None:
        """添加一个检索 Provider
        
        Args:
            provider: 检索 Provider 实例
        """
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.name}")

    @classmethod
    def from_single_store(
        cls,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        storage_role: StorageRole = None
    ) -> Retriever:
        """从单个向量存储创建 Retriever（能力自适应）
        
        自动根据 VDB 能力选择最佳检索策略。
        
        Args:
            embedder: 嵌入器
            vector_store: 向量存储
            storage_role: 存储角色（可选，用于命名）
            
        Returns:
            配置好的 Retriever 实例
        """
        provider = ProviderFactory.create_for_store(embedder, vector_store, storage_role)
        return cls(embedder=embedder, providers=[provider])

    @classmethod
    def from_multi_stores(
        cls,
        embedder: BaseEmbedder,
        stores_config: list[tuple[BaseVectorStore, StorageRole]],
        fusion_strategy: str = "rrf",
        fusion_weights: list[float] | None = None
    ) -> Retriever:
        """从多个向量存储创建 Retriever（角色分工）
        
        支持场景：
        - Chroma (vector_only) + DuckDB (fulltext_only)
        - SeekDB (primary) + Chroma (backup)
        - 任意组合
        
        Args:
            embedder: 嵌入器
            stores_config: [(vector_store, role), ...] 列表
            fusion_strategy: 融合策略
            fusion_weights: Provider 权重
            
        Returns:
            配置好的 Retriever 实例
        """
        logger.info(f"Creating Retriever from {len(stores_config)} stores")
        
        providers = []
        for vector_store, role in stores_config:
            provider = ProviderFactory.create_for_role(embedder, vector_store, role)
            if provider:
                providers.append(provider)
        
        if not providers:
            raise ValueError("No valid providers created from stores config")
        
        return cls(
            embedder=embedder,
            providers=providers,
            fusion_strategy=fusion_strategy,
            fusion_weights=fusion_weights
        )

    async def retrieve(
        self,
        query: str,
        top_k: int
    ) -> list[SearchResult]:
        """执行检索
        
        如果有多个 Provider，会并行检索并融合结果。
        如果只有一个 Provider，直接返回其结果。
        
        Args:
            query: 查询文本
            top_k: 最终返回的结果数
            
        Returns:
            检索结果列表
        """
        if not self.providers:
            logger.warning("No providers configured")
            return []
        
        logger.info(f"Retrieving for query: {query[:50]}...")
        
        # 单 Provider：直接检索
        if len(self.providers) == 1:
            return await self.providers[0].retrieve(query, top_k)
        
        # 多 Provider：并行检索 + 融合
        return await self._multi_provider_retrieve(query, top_k)

    async def _multi_provider_retrieve(
        self,
        query: str,
        top_k: int
    ) -> list[SearchResult]:
        """多 Provider 并行检索并融合"""
        # 每个 Provider 多检索一些候选
        candidate_k = max(top_k * 2, 20)
        
        logger.info(
            f"Parallel retrieval from {len(self.providers)} providers "
            f"(candidate_k={candidate_k})"
        )
        
        # 并行检索
        tasks = [p.retrieve(query, candidate_k) for p in self.providers]
        results_list = await asyncio.gather(*tasks)
        
        # 打印每个 Provider 的结果数
        for i, results in enumerate(results_list):
            logger.debug(
                f"  Provider {i} ({self.providers[i].name}): "
                f"{len(results)} results"
            )
        
        # 融合结果
        if self.fusion_strategy == "weighted_rrf" and self.fusion_weights:
            fused = weighted_rrf(
                results_list,
                weights=self.fusion_weights,
                k=self.rrf_k,
                top_k=top_k
            )
            logger.info(f"Weighted RRF fusion: {len(fused)} results")
        else:
            fused = reciprocal_rank_fusion(
                results_list,
                k=self.rrf_k,
                top_k=top_k
            )
            logger.info(f"Standard RRF fusion: {len(fused)} results")
        
        return fused

