"""SQLModel database models for business logic."""

from datetime import datetime

from sqlmodel import Field, SQLModel


class KnowledgeBase(SQLModel, table=True):
    """Knowledge Base Metadata Table"""
    __tablename__ = "knowledge_bases"

    id: int | None = Field(default=None, primary_key=True)
    kb_id: str = Field(index=True, unique=True)  # Business ID
    name: str
    description: str | None = None
    vdb_type: str = "chroma"  # chroma, duckdb, seekdb
    embedder_name: str | None = None  # Associated embedder configuration name
    collection_name: str  # Vector store collection name
    indexing_technique: str = "high_quality"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # ========== 检索配置 (Retrieval Configuration) ==========
    # 搜索模式: "hybrid", "vector", "keyword"
    search_mode: str = "hybrid"
    # 默认返回结果数量
    top_k: int = 5
    # 分数阈值，低于此分数的结果将被过滤
    score_threshold: float = 0.0
    
    # Reranker 配置
    reranker_enabled: bool = False
    reranker_type: str | None = None  # "cohere", "qwen", "noop" 等
    reranker_model: str | None = None  # 具体模型名，如 "rerank-english-v3.0"
    reranker_api_key: str | None = None  # Reranker API Key
    reranker_top_k: int | None = None  # Rerank 后返回的数量，None 表示使用 top_k
    
    # Query Rewriter 配置
    rewriter_enabled: bool = False
    rewriter_llm_name: str | None = None  # 使用的 LLM 配置名称
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Document(SQLModel, table=True):
    """Document Metadata Table"""
    __tablename__ = "documents"

    id: int | None = Field(default=None, primary_key=True)
    kb_id: str = Field(index=True)  # Associated knowledge base
    filename: str
    file_size: int  # bytes
    chunk_count: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None


class EmbedderConfig(SQLModel, table=True):
    """Embedding Model Configuration Table"""
    __tablename__ = "embedder_configs"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)  # Configuration name
    embedder_type: str = "openai"  # openai, seekdb
    base_url: str = ""  # Base URL for OpenAI-compatible API (can be empty for seekdb type)
    api_key: str = ""  # API Key (can be empty for seekdb built-in models)
    model: str  # Model name
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LLMConfig(SQLModel, table=True):
    """LLM Model Configuration Table (OpenAI-compatible interface)"""
    __tablename__ = "llm_configs"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)  # Configuration name (e.g., Kimi, GPT-4)
    base_url: str = "https://api.moonshot.cn/v1"  # Default Base URL
    api_key: str = ""
    model: str = "kimi-k2-turbo-preview"  # Model name
    model_path: str | None = None # Path to local model file (for local LLMs)
    temperature: float = 0.7
    max_tokens: int = 2048
    is_active: bool = False  # Whether this is the current default conversation model
    created_at: datetime = Field(default_factory=datetime.utcnow)
