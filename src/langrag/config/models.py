"""Configuration models for RAG components.

This module defines Pydantic models for component configuration.
All components are configured via a type string and optional parameters.
"""

from typing import Any
from enum import Enum
from pydantic import BaseModel, Field


class StorageRole(str, Enum):
    """存储角色定义，用于多数据源场景
    
    Attributes:
        PRIMARY: 主存储，存储完整数据（chunks + embeddings + metadata）
        VECTOR_ONLY: 仅向量存储，只存储 embeddings 和基本元数据
        FULLTEXT_ONLY: 仅全文存储，只存储文本内容用于关键词检索
        BACKUP: 备份存储，完整冗余备份
    """
    PRIMARY = "primary"
    VECTOR_ONLY = "vector_only"
    FULLTEXT_ONLY = "fulltext_only"
    BACKUP = "backup"


class ComponentConfig(BaseModel):
    """Configuration for a single component.

    Attributes:
        type: Component type identifier (e.g., "simple_text", "fixed_size")
        params: Component-specific parameters as a dictionary
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(ComponentConfig):
    """扩展的向量存储配置，支持存储角色定义
    
    Attributes:
        type: 存储类型（如 "seekdb", "chroma", "duckdb"）
        params: 存储参数
        role: 存储角色（用于多数据源场景）
        enabled: 是否启用该存储（用于动态切换）
    """
    role: StorageRole = StorageRole.PRIMARY
    enabled: bool = True


class RetrievalConfig(BaseModel):
    """检索配置
    
    Attributes:
        mode: 检索模式 ("single" | "multi_store" | "auto")
        fusion_strategy: 多源融合策略 ("rrf" | "weighted_rrf" | "linear")
        fusion_weights: 融合权重（用于 weighted_rrf）
        top_k_per_store: 每个存储检索的结果数
        final_top_k: 最终返回的结果数
    """
    mode: str = "auto"  # "single" | "multi_store" | "auto"
    fusion_strategy: str = "rrf"  # "rrf" | "weighted_rrf" | "linear"
    fusion_weights: list[float] | None = None
    top_k_per_store: int = 10
    final_top_k: int = 5


class RAGConfig(BaseModel):
    """Main RAG system configuration.

    Attributes:
        parser: Parser component configuration
        chunker: Chunker component configuration
        embedder: Embedder component configuration
        vector_store: 单一向量存储配置（向后兼容）
        vector_stores: 多向量存储配置（新增）
        reranker: Optional reranker component configuration
        llm: Optional LLM component configuration
        retrieval: 检索配置
        retrieval_top_k: Number of results to retrieve from vector search (deprecated)
        rerank_top_k: Number of results to return after reranking (deprecated)
    """

    parser: ComponentConfig
    chunker: ComponentConfig
    embedder: ComponentConfig
    
    # 向量存储配置：支持单一或多个
    vector_store: VectorStoreConfig | None = None  # 单一存储（向后兼容）
    vector_stores: list[VectorStoreConfig] | None = None  # 多存储（新增）
    
    reranker: ComponentConfig | None = None
    llm: ComponentConfig | None = None

    # 检索配置
    retrieval: RetrievalConfig | None = None

    # Pipeline settings (deprecated, use retrieval config instead)
    retrieval_top_k: int = Field(default=5, ge=1)
    rerank_top_k: int | None = Field(default=None, ge=1)

    model_config = {
        "extra": "allow",  # Allow additional fields for extensibility
    }

    def get_vector_stores(self) -> list[VectorStoreConfig]:
        """获取所有向量存储配置（统一处理单一和多个的情况）
        
        Returns:
            向量存储配置列表
        """
        if self.vector_stores:
            return [vs for vs in self.vector_stores if vs.enabled]
        elif self.vector_store:
            return [self.vector_store]
        return []
    
    def get_retrieval_config(self) -> RetrievalConfig:
        """获取检索配置（兼容旧配置）
        
        Returns:
            检索配置对象
        """
        if self.retrieval:
            return self.retrieval
        
        # 向后兼容：从旧配置构建
        return RetrievalConfig(
            mode="auto",
            final_top_k=self.rerank_top_k or self.retrieval_top_k,
            top_k_per_store=self.retrieval_top_k,
        )
