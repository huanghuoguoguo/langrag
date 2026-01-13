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

# Errors
from .errors import (
    LangRAGError,
    RetryableError,
    RateLimitError,
    ServiceUnavailableError,
    ConnectionError,
    TransientError,
    TimeoutError,
    ConnectTimeoutError,
    ReadTimeoutError,
    PermanentError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    ConfigurationError,
    QuotaExceededError,
    EmbeddingError,
    RetrievalError,
    IndexingError,
    VectorStoreError,
    is_retryable,
    classify_http_error,
    wrap_exception,
)

# Evaluation
from .evaluation import (
    BaseEvaluator,
    EvaluationRunner,
    EvaluationSample,
    EvaluationResult,
    EvaluationReport,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    evaluate_rag,
)

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
from .index_processor.processor.raptor import RaptorConfig, RaptorIndexProcessor
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

    # Errors
    "LangRAGError",
    "RetryableError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ConnectionError",
    "TransientError",
    "TimeoutError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "PermanentError",
    "AuthenticationError",
    "InvalidRequestError",
    "NotFoundError",
    "ConfigurationError",
    "QuotaExceededError",
    "EmbeddingError",
    "RetrievalError",
    "IndexingError",
    "VectorStoreError",
    "is_retryable",
    "classify_http_error",
    "wrap_exception",

    # Evaluation
    "BaseEvaluator",
    "EvaluationRunner",
    "EvaluationSample",
    "EvaluationResult",
    "EvaluationReport",
    "FaithfulnessEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextRelevancyEvaluator",
    "evaluate_rag",

    # Indexing
    "BaseParser",
    "SimpleTextParser",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "BaseIndexProcessor",
    "ParagraphIndexProcessor",
    "QAIndexProcessor",
    "ParentChildIndexProcessor",
    "RaptorIndexProcessor",
    "RaptorConfig",

    # Retrieval
    "RetrievalWorkflow",
    "RetrievalService",

    # LLM
    "BaseEmbedder",
    "EmbedderFactory",

    # Data Source
    "BaseVector",
]
