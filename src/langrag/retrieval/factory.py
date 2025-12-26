"""Provider 工厂 - 根据 VectorStore 能力自动创建最佳 Provider"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .providers.fulltext import FullTextSearchProvider
from .providers.hybrid import HybridSearchProvider
from .providers.vector import VectorSearchProvider

if TYPE_CHECKING:
    from ..config.models import StorageRole
    from ..embedder import BaseEmbedder
    from ..vector_store import BaseVectorStore
    from .base import BaseRetrievalProvider


class ProviderFactory:
    """Provider 工厂

    根据 VectorStore 的能力自动选择最佳的检索策略：
    - hybrid > (vector + fulltext) > vector > fulltext
    """

    # 策略优先级映射
    STRATEGY_PRIORITY = [
        # (条件检查函数, Provider类, 描述)
        (lambda caps: caps.supports_hybrid, HybridSearchProvider, "native hybrid search"),
        (
            lambda caps: caps.supports_vector and caps.supports_fulltext,
            HybridSearchProvider,
            "RRF-fused hybrid search (vector + fulltext)",
        ),
        (lambda caps: caps.supports_vector, VectorSearchProvider, "vector-only search"),
        (lambda caps: caps.supports_fulltext, FullTextSearchProvider, "fulltext-only search"),
    ]

    @classmethod
    def create_for_store(
        cls, embedder: BaseEmbedder, vector_store: BaseVectorStore, role: StorageRole | None = None
    ) -> BaseRetrievalProvider:
        """为 VectorStore 创建最佳 Provider

        Args:
            embedder: 嵌入器
            vector_store: 向量存储
            role: 存储角色（可选，用于命名）

        Returns:
            配置好的 Provider 实例

        Raises:
            ValueError: 如果 VectorStore 不支持任何检索模式
        """
        caps = vector_store.capabilities
        store_name = vector_store.__class__.__name__
        provider_name = f"{store_name}({role.value})" if role else store_name

        # 按优先级查找匹配的策略
        for check_fn, provider_class, description in cls.STRATEGY_PRIORITY:
            if check_fn(caps):
                # 创建 Provider
                if provider_class == FullTextSearchProvider:
                    # FullTextSearchProvider 不需要 embedder
                    provider = provider_class(vector_store=vector_store, name=provider_name)
                else:
                    provider = provider_class(
                        embedder=embedder, vector_store=vector_store, name=provider_name
                    )

                logger.info(
                    f"Created {provider.__class__.__name__} for {store_name}: {description}"
                )
                return provider

        # 没有匹配的策略
        raise ValueError(
            f"VectorStore '{store_name}' doesn't support any search mode. Capabilities: {caps}"
        )

    @classmethod
    def create_for_role(
        cls, embedder: BaseEmbedder, vector_store: BaseVectorStore, role: StorageRole
    ) -> BaseRetrievalProvider | None:
        """根据角色为 VectorStore 创建 Provider

        Args:
            embedder: 嵌入器
            vector_store: 向量存储
            role: 存储角色

        Returns:
            Provider 实例，如果角色为 BACKUP 则返回 None
        """
        from ..config.models import StorageRole as Role

        store_name = vector_store.__class__.__name__
        caps = vector_store.capabilities
        provider_name = f"{store_name}({role.value})"

        # BACKUP 角色不参与检索
        if role == Role.BACKUP:
            logger.info(f"Skipping BACKUP store: {store_name}")
            return None

        # VECTOR_ONLY 角色：只创建向量 Provider
        if role == Role.VECTOR_ONLY:
            if not caps.supports_vector:
                logger.warning(f"VECTOR_ONLY store '{store_name}' doesn't support vector search")
                return None
            return VectorSearchProvider(
                embedder=embedder, vector_store=vector_store, name=provider_name
            )

        # FULLTEXT_ONLY 角色：只创建全文 Provider
        if role == Role.FULLTEXT_ONLY:
            if not caps.supports_fulltext:
                logger.warning(
                    f"FULLTEXT_ONLY store '{store_name}' doesn't support fulltext search"
                )
                return None
            return FullTextSearchProvider(vector_store=vector_store, name=provider_name)

        # PRIMARY 角色：使用最佳可用策略
        if role == Role.PRIMARY:
            try:
                return cls.create_for_store(embedder, vector_store, role)
            except ValueError as e:
                logger.warning(f"Cannot create provider for PRIMARY store '{store_name}': {e}")
                return None

        # 未知角色
        logger.warning(f"Unknown storage role: {role}")
        return None
