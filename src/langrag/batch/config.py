"""
Batch processing configuration.

This module defines configuration options for batch processing operations,
allowing fine-grained control over batch sizes, parallelism, and behavior.
"""

from dataclasses import dataclass, field


@dataclass
class BatchConfig:
    """
    Configuration for batch processing operations.

    This configuration controls how documents are processed in batches,
    including embedding generation, vector store insertion, and error handling.

    Attributes:
        embedding_batch_size: Number of texts to embed in a single API call.
            Larger batches are more efficient but may hit API limits.
            Default: 100 (suitable for most embedding APIs)

        storage_batch_size: Number of documents to insert into vector store
            in a single batch. Larger batches reduce overhead but use more memory.
            Default: 500

        max_retries: Maximum number of retry attempts for failed operations.
            Default: 3

        retry_delay: Initial delay between retries in seconds.
            Uses exponential backoff (delay * 2^attempt).
            Default: 1.0

        continue_on_error: Whether to continue processing after an error.
            If True, failed documents are logged and skipped.
            If False, processing stops on first error.
            Default: False

        show_progress: Whether to emit progress callbacks.
            Default: True

    Example:
        >>> config = BatchConfig(
        ...     embedding_batch_size=50,  # Smaller batches for limited API
        ...     storage_batch_size=1000,  # Larger batches for fast DB
        ...     continue_on_error=True    # Don't stop on failures
        ... )
    """

    embedding_batch_size: int = 100
    storage_batch_size: int = 500
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = False
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if self.storage_batch_size < 1:
            raise ValueError("storage_batch_size must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
