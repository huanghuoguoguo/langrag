"""
LangRAG - A modular Retrieval-Augmented Generation framework.

This package provides a clean, extensible architecture for building RAG systems
with pluggable components and type-based configuration.
"""

__version__ = "0.1.0"

# Core entities
from .core.document import Document
from .core.chunk import Chunk
from .core.query import Query
from .core.search_result import SearchResult

# Modular components
from .parser import BaseParser, SimpleTextParser, ParserFactory
from .chunker import BaseChunker, FixedSizeChunker, RecursiveCharacterChunker, ChunkerFactory
from .embedder import BaseEmbedder, MockEmbedder, EmbedderFactory
from .vector_store import (
    BaseVectorStore,
    VectorStoreCapabilities,
    SearchMode,
    InMemoryVectorStore,
    VectorStoreFactory,
    VectorStoreManager
)
from .reranker import BaseReranker, NoOpReranker, RerankerFactory
from .llm import BaseLLM, LLMFactory

# Configuration
from .config.models import RAGConfig, ComponentConfig, StorageRole
from .config.factory import ComponentFactory

# Knowledge base management
from .knowledge import KnowledgeBase, KnowledgeBaseManager

# Pipelines
from .indexing import IndexingPipeline

# Retrieval system
from .retrieval import (
    Retriever,
    BaseRetrievalProvider,
    VectorSearchProvider,
    FullTextSearchProvider,
    HybridSearchProvider
)

# Engine (high-level orchestrator)
from .engine import RAGEngine

# Utilities
from .utils import cosine_similarity, reciprocal_rank_fusion, weighted_rrf

__all__ = [
    # Version
    "__version__",
    # Core
    "Document",
    "Chunk",
    "Query",
    "SearchResult",
    # Parser
    "BaseParser",
    "SimpleTextParser",
    "ParserFactory",
    # Chunker
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "ChunkerFactory",
    # Embedder
    "BaseEmbedder",
    "MockEmbedder",
    "EmbedderFactory",
    # Vector Store
    "BaseVectorStore",
    "VectorStoreCapabilities",
    "SearchMode",
    "InMemoryVectorStore",
    "VectorStoreFactory",
    "VectorStoreManager",
    # Reranker
    "BaseReranker",
    "NoOpReranker",
    "RerankerFactory",
    # LLM
    "BaseLLM",
    "LLMFactory",
    # Config
    "RAGConfig",
    "ComponentConfig",
    "StorageRole",
    "ComponentFactory",
    # Knowledge Base
    "KnowledgeBase",
    "KnowledgeBaseManager",
    # Pipelines
    "IndexingPipeline",
    # Retrieval system
    "Retriever",
    "BaseRetrievalProvider",
    "VectorSearchProvider",
    "FullTextSearchProvider",
    "HybridSearchProvider",
    # Engine
    "RAGEngine",
    # Utilities
    "cosine_similarity",
    "reciprocal_rank_fusion",
    "weighted_rrf",
]


def hello() -> str:
    """Legacy function - kept for backward compatibility."""
    return "Hello from langrag!"
