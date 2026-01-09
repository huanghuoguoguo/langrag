"""
LangRAG - A modular Retrieval-Augmented Generation framework.
"""

__version__ = "0.2.0"

# Entities
from .datasource.service import RetrievalService

# Batch Processing
from .batch import BatchConfig, BatchProcessor, BatchProgress, ProgressCallback

# Cache
from .cache import BaseCache, CacheEntry, SemanticCache

# Data Source
from .datasource.vdb.base import BaseVector
from .entities.dataset import Dataset, RetrievalContext
from .entities.document import Document, DocumentType
from .entities.search_result import SearchResult

# Indexing Components
from .index_processor.extractor.base import BaseParser
from .index_processor.extractor.providers.simple_text import SimpleTextParser
from .index_processor.processor.base import BaseIndexProcessor
from .index_processor.processor.paragraph import ParagraphIndexProcessor
from .index_processor.processor.parent_child import ParentChildIndexProcessor
from .index_processor.processor.qa import QAIndexProcessor
from .index_processor.splitter.base import BaseChunker
from .index_processor.splitter.providers.recursive_character import RecursiveCharacterChunker

# LLM & Embedding
from .llm.embedder.base import BaseEmbedder
from .llm.embedder.factory import EmbedderFactory

# Retrieval Components
from .retrieval.workflow import RetrievalWorkflow

__all__ = [
    # Entities
    "Document",
    "DocumentType",
    "Dataset",
    "RetrievalContext",
    "SearchResult",

    # Batch Processing
    "BatchConfig",
    "BatchProcessor",
    "BatchProgress",
    "ProgressCallback",

    # Cache
    "BaseCache",
    "CacheEntry",
    "SemanticCache",

    # Indexing
    "BaseParser",
    "SimpleTextParser",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "BaseIndexProcessor",
    "ParagraphIndexProcessor",
    "QAIndexProcessor",
    "ParentChildIndexProcessor",

    # Retrieval
    "RetrievalWorkflow",
    "RetrievalService",

    # LLM
    "BaseEmbedder",
    "EmbedderFactory",

    # Data Source
    "BaseVector",
]
