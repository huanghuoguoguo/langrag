"""SQLModel database models for business logic."""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class KnowledgeBase(SQLModel, table=True):
    """知识库元数据表"""
    __tablename__ = "knowledge_bases"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    kb_id: str = Field(index=True, unique=True)  # 业务 ID
    name: str
    description: Optional[str] = None
    vdb_type: str = "chroma"  # chroma, duckdb, seekdb
    embedder_name: Optional[str] = None  # 关联的 embedder 配置名称
    collection_name: str  # 向量库集合名
    indexing_technique: str = "high_quality"
    chunk_size: int = 500
    chunk_overlap: int = 50
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Document(SQLModel, table=True):
    """文档元数据表"""
    __tablename__ = "documents"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    kb_id: str = Field(index=True)  # 关联知识库
    filename: str
    file_size: int  # bytes
    chunk_count: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class EmbedderConfig(SQLModel, table=True):
    """Embedding 模型配置表"""
    __tablename__ = "embedder_configs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)  # 配置名称
    embedder_type: str = "openai"  # openai, seekdb
    base_url: str = ""  # OpenAI 兼容 API 的 base URL（seekdb 类型时可为空）
    api_key: str = ""  # API Key（seekdb 内置模型时可为空）
    model: str  # 模型名称
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LLMConfig(SQLModel, table=True):
    """LLM 模型配置表 (OpenAI 兼容接口)"""
    __tablename__ = "llm_configs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)  # 配置名称 (如 Kimi, GPT-4)
    base_url: str = "https://api.moonshot.cn/v1"  # 默认 Base URL
    api_key: str
    model: str = "kimi-k2-turbo-preview"  # 模型名称
    temperature: float = 0.7
    max_tokens: int = 2048
    is_active: bool = False  # 是否为当前默认对话模型
    created_at: datetime = Field(default_factory=datetime.utcnow)
