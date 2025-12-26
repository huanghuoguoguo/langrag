"""
LangRAG - A modular Retrieval-Augmented Generation framework.

This package provides a clean, extensible architecture for building RAG systems
with pluggable components and type-based configuration.
"""

__version__ = "0.1.0"

# Core entities
from .chunker import BaseChunker, ChunkerFactory, FixedSizeChunker, RecursiveCharacterChunker
from .compressor import BaseCompressor, CompressorFactory
from .config.factory import ComponentFactory

# Configuration
from .config.models import ComponentConfig, RAGConfig, StorageRole
from .core.chunk import Chunk
from .core.document import Document
from .core.query import Query
from .core.search_result import SearchResult
from .embedder import BaseEmbedder, EmbedderFactory, MockEmbedder

# Engine (high-level orchestrator)
from .engine import RAGEngine

# Pipelines
from .indexing import IndexingPipeline

# Knowledge base management
from .knowledge import KnowledgeBase, KnowledgeBaseManager
from .llm import BaseLLM, LLMFactory

# Modular components
from .parser import BaseParser, ParserFactory, SimpleTextParser
from .reranker import BaseReranker, NoOpReranker, RerankerFactory

# Retrieval system
from .retrieval import (
    BaseRetrievalProvider,
    FullTextSearchProvider,
    HybridSearchProvider,
    Retriever,
    VectorSearchProvider,
)

# Utilities
from .utils import cosine_similarity, reciprocal_rank_fusion, weighted_rrf
from .vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    SearchMode,
    VectorStoreCapabilities,
    VectorStoreFactory,
    VectorStoreManager,
)

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
    # Compressor
    "BaseCompressor",
    "CompressorFactory",
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
