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
from .vector_store import BaseVectorStore, InMemoryVectorStore, VectorStoreFactory
from .reranker import BaseReranker, NoOpReranker, RerankerFactory
from .llm import BaseLLM, LLMFactory

# Configuration
from .config.models import RAGConfig, ComponentConfig
from .config.factory import ComponentFactory

# Pipelines
from .pipeline.indexing import IndexingPipeline
from .pipeline.retrieval import RetrievalPipeline

# Engine (high-level orchestrator)
from .engine import RAGEngine

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
    "InMemoryVectorStore",
    "VectorStoreFactory",
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
    "ComponentFactory",
    # Pipelines
    "IndexingPipeline",
    "RetrievalPipeline",
    # Engine
    "RAGEngine",
]


def hello() -> str:
    """Legacy function - kept for backward compatibility."""
    return "Hello from langrag!"
