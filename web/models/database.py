"""SQLModel database models for business logic."""

from datetime import datetime

from sqlmodel import Field, SQLModel


class KnowledgeBase(SQLModel, table=True):
    """Knowledge Base Metadata Table

    Only stores **offline/indexing** configuration that cannot be changed after creation.
    Retrieval configuration (reranker, rewriter, top_k, etc.) is passed dynamically at query time.
    """
    __tablename__ = "knowledge_bases"

    id: int | None = Field(default=None, primary_key=True)
    kb_id: str = Field(index=True, unique=True)  # Business ID
    name: str
    description: str | None = None

    # ========== Offline/Indexing Configuration (Immutable after creation) ==========
    vdb_type: str = "chroma"  # chroma, duckdb, seekdb
    embedder_name: str | None = None  # Associated embedder configuration name
    collection_name: str  # Vector store collection name

    # Indexing strategy: paragraph (default), qa, raptor
    indexing_technique: str = "paragraph"
    # LLM for QA/RAPTOR indexing (selected from LLM pool)
    indexing_llm_name: str | None = None

    chunk_size: int = 500
    chunk_overlap: int = 50

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
