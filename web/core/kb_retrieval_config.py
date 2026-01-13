"""
知识库检索配置数据类

每个知识库可以拥有独立的检索配置，包括：
- 搜索模式 (hybrid, vector, keyword)
- Reranker 配置
- Query Rewriter 配置
- 结果过滤参数
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankerConfig:
    """Reranker 配置"""
    enabled: bool = False
    reranker_type: str | None = None  # "cohere", "qwen", "noop"
    model: str | None = None  # 具体模型名
    api_key: str | None = None
    top_k: int | None = None  # Rerank 后返回数量，None 表示使用默认 top_k
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reranker_type": self.reranker_type,
            "model": self.model,
            "api_key": self.api_key,
            "top_k": self.top_k
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RerankerConfig":
        return cls(
            enabled=data.get("enabled", False),
            reranker_type=data.get("reranker_type"),
            model=data.get("model"),
            api_key=data.get("api_key"),
            top_k=data.get("top_k")
        )


@dataclass
class RewriterConfig:
    """Query Rewriter 配置"""
    enabled: bool = False
    llm_name: str | None = None  # 使用的 LLM 配置名称
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "llm_name": self.llm_name
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewriterConfig":
        return cls(
            enabled=data.get("enabled", False),
            llm_name=data.get("llm_name")
        )


@dataclass
class KBRetrievalConfig:
    """
    知识库检索配置
    
    封装了单个知识库的完整检索配置，包括搜索模式、reranker、rewriter 等。
    """
    kb_id: str
    
    # 搜索配置
    search_mode: str = "hybrid"  # "hybrid", "vector", "keyword"
    top_k: int = 5
    score_threshold: float = 0.0
    
    # 组件配置
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    
    # Embedder 配置（可选，None 表示使用全局 Embedder）
    embedder_name: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "search_mode": self.search_mode,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "reranker": self.reranker.to_dict(),
            "rewriter": self.rewriter.to_dict(),
            "embedder_name": self.embedder_name
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBRetrievalConfig":
        return cls(
            kb_id=data["kb_id"],
            search_mode=data.get("search_mode", "hybrid"),
            top_k=data.get("top_k", 5),
            score_threshold=data.get("score_threshold", 0.0),
            reranker=RerankerConfig.from_dict(data.get("reranker", {})),
            rewriter=RewriterConfig.from_dict(data.get("rewriter", {})),
            embedder_name=data.get("embedder_name")
        )
    
    @classmethod
    def from_kb_model(cls, kb) -> "KBRetrievalConfig":
        """
        从 KnowledgeBase 数据库模型创建配置
        
        Args:
            kb: KnowledgeBase 模型实例
        """
        return cls(
            kb_id=kb.kb_id,
            search_mode=kb.search_mode or "hybrid",
            top_k=kb.top_k or 5,
            score_threshold=kb.score_threshold or 0.0,
            reranker=RerankerConfig(
                enabled=kb.reranker_enabled or False,
                reranker_type=kb.reranker_type,
                model=kb.reranker_model,
                api_key=kb.reranker_api_key,
                top_k=kb.reranker_top_k
            ),
            rewriter=RewriterConfig(
                enabled=kb.rewriter_enabled or False,
                llm_name=kb.rewriter_llm_name
            ),
            embedder_name=kb.embedder_name
        )
