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
            Uses exponential backoff (delay * exponential_base^attempt).
            Default: 1.0

        retry_max_delay: Maximum delay between retries in seconds.
            Default: 60.0

        retry_exponential_base: Base for exponential backoff calculation.
            Default: 2.0

        retry_jitter: Random jitter factor (0-1) to add to delays.
            Helps prevent thundering herd problem.
            Default: 0.1

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
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0
    retry_jitter: float = 0.1
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
        if self.retry_max_delay < self.retry_delay:
            raise ValueError("retry_max_delay must be >= retry_delay")
        if not 0 <= self.retry_jitter <= 1:
            raise ValueError("retry_jitter must be between 0 and 1")
