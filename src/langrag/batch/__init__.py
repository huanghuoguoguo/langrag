"""
Batch Processing Module for LangRAG.

This module provides utilities for processing large volumes of documents
efficiently with features like:
- Configurable batch sizes for embedding and storage
- Progress tracking via callbacks
- Error handling with retry logic
- Memory-efficient streaming

Example:
    >>> from langrag.batch import BatchProcessor, BatchConfig
    >>> processor = BatchProcessor(embedder, vector_store)
    >>> processor.process_documents(
    ...     documents,
    ...     config=BatchConfig(embedding_batch_size=100),
    ...     on_progress=lambda p: print(f"{p.percent:.0%}")
    ... )
"""

from langrag.batch.config import BatchConfig
from langrag.batch.processor import BatchProcessor
from langrag.batch.progress import BatchProgress, ProgressCallback

__all__ = [
    "BatchConfig",
    "BatchProcessor",
    "BatchProgress",
    "ProgressCallback",
]
