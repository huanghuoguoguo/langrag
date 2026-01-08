"""
LangRAG - A modular Retrieval-Augmented Generation framework.
"""

__version__ = "0.2.0"

# Entities
from .entities.document import Document, DocumentType
from .entities.dataset import Dataset, RetrievalContext
from .entities.search_result import SearchResult

# Indexing Components
from .index_processor.extractor.base import BaseParser
from .index_processor.extractor.providers.simple_text import SimpleTextParser
from .index_processor.splitter.base import BaseChunker
from .index_processor.splitter.providers.recursive_character import RecursiveCharacterChunker
from .index_processor.processor.base import BaseIndexProcessor
from .index_processor.processor.paragraph import ParagraphIndexProcessor
from .index_processor.processor.qa import QAIndexProcessor

# Retrieval Components
from .retrieval.workflow import RetrievalWorkflow
from .datasource.service import RetrievalService

# LLM & Embedding
from .llm.embedder.base import BaseEmbedder
from .llm.embedder.factory import EmbedderFactory

# Data Source
from .datasource.vdb.base import BaseVector

__all__ = [
    # Entities
    "Document",
    "DocumentType",
    "Dataset",
    "RetrievalContext",
    "SearchResult",
    
    # Indexing
    "BaseParser",
    "SimpleTextParser",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "BaseIndexProcessor",
    "ParagraphIndexProcessor",
    "QAIndexProcessor",
    
    # Retrieval
    "RetrievalWorkflow",
    "RetrievalService",
    
    # LLM
    "BaseEmbedder",
    "EmbedderFactory",
    
    # Data Source
    "BaseVector",
]
