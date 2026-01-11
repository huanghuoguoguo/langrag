"""
Batch processor for efficient document processing.

This module provides the main BatchProcessor class that handles
efficient processing of large document collections with:
- Configurable batch sizes for embedding and storage
- Progress tracking via callbacks
- Error handling with exponential backoff retry
- Memory-efficient chunked processing
- Comprehensive logging for debugging
"""

import logging
import time
import uuid
from typing import Any

from langrag.batch.config import BatchConfig
from langrag.batch.progress import (
    BatchProgress,
    BatchStage,
    CallbackProgressReporter,
    ProgressCallback,
)
from langrag.datasource.vdb.base import BaseVector
from langrag.entities.document import Document
from langrag.errors import (
    EmbeddingError,
    IndexingError,
    is_retryable,
)
from langrag.llm.embedder.base import BaseEmbedder
from langrag.utils.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Efficient batch processor for document embedding and storage.

    This processor handles the embedding and storage of large document
    collections by processing them in configurable batches, with support
    for progress tracking, error handling, and automatic retry.

    The processing flow:
    1. Split documents into embedding batches
    2. Generate embeddings for each batch (with retry on transient errors)
    3. Accumulate embedded documents
    4. Store in storage batches when threshold reached
    5. Report progress throughout

    Features:
    - Exponential backoff with jitter for transient errors
    - Respects rate limit headers (Retry-After)
    - Request ID tracking for log correlation
    - Detailed statistics including timing and error tracking

    Attributes:
        embedder: The embedding model to use
        vector_store: Target vector store for documents
        config: Batch processing configuration

    Example:
        >>> processor = BatchProcessor(embedder, vector_store)
        >>> stats = processor.process_documents(
        ...     documents,
        ...     on_progress=lambda p: print(f"{p.percent:.0%}")
        ... )
        >>> print(f"Processed {stats['total']} docs with {stats['errors']} errors")
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVector,
        config: BatchConfig | None = None
    ):
        """
        Initialize the batch processor.

        Args:
            embedder: Embedding model for generating vectors
            vector_store: Vector store for document storage
            config: Batch configuration (uses defaults if not provided)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or BatchConfig()

        # Build retry config from batch config
        self._retry_config = RetryConfig(
            max_attempts=self.config.max_retries + 1,  # +1 because max_retries is additional attempts
            base_delay=self.config.retry_delay,
            max_delay=self.config.retry_max_delay,
            exponential_base=self.config.retry_exponential_base,
            jitter=self.config.retry_jitter,
        )

    def process_documents(
        self,
        documents: list[Document],
        on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """
        Process a list of documents through embedding and storage.

        This method processes documents in batches, generating embeddings
        and storing them in the vector store. Progress is reported via
        the optional callback.

        Args:
            documents: List of documents to process (must have page_content)
            on_progress: Optional callback for progress updates

        Returns:
            Dictionary with processing statistics:
            - total: Total documents processed
            - embedded: Documents successfully embedded
            - stored: Documents successfully stored
            - errors: Number of errors encountered
            - duration: Total processing time in seconds
            - request_id: Unique identifier for this batch job

        Raises:
            Exception: If continue_on_error is False and an error occurs
        """
        # Generate request ID for log correlation
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        stats = {
            "total": len(documents),
            "embedded": 0,
            "stored": 0,
            "errors": 0,
            "duration": 0.0,
            "request_id": request_id,
            "embedding_time": 0.0,
            "storage_time": 0.0,
        }

        if not documents:
            logger.debug(f"[{request_id}] No documents to process")
            return stats

        reporter = CallbackProgressReporter(on_progress) if on_progress else None
        embedded_docs: list[Document] = []
        total_docs = len(documents)

        # Calculate batch counts
        embed_batch_size = self.config.embedding_batch_size
        store_batch_size = self.config.storage_batch_size
        total_embed_batches = (total_docs + embed_batch_size - 1) // embed_batch_size

        logger.info(
            f"[{request_id}] Batch processing started: "
            f"documents={total_docs}, embed_batch_size={embed_batch_size}, "
            f"store_batch_size={store_batch_size}, max_retries={self.config.max_retries}"
        )

        # Process embedding batches
        for batch_idx in range(0, total_docs, embed_batch_size):
            batch_num = batch_idx // embed_batch_size + 1
            batch_end = min(batch_idx + embed_batch_size, total_docs)
            batch_docs = documents[batch_idx:batch_end]

            # Report progress
            if reporter and self.config.show_progress:
                reporter.report(BatchProgress.create(
                    stage=BatchStage.EMBEDDING,
                    current=batch_idx,
                    total=total_docs,
                    message=f"Embedding batch {batch_num}/{total_embed_batches}",
                    errors=stats["errors"],
                    batch_num=batch_num,
                    total_batches=total_embed_batches
                ))

            # Embed batch with retry
            embed_start = time.time()
            try:
                batch_with_embeddings = self._embed_batch(batch_docs, request_id, batch_num)
                embedded_docs.extend(batch_with_embeddings)
                stats["embedded"] += len(batch_with_embeddings)
                stats["embedding_time"] += time.time() - embed_start

                logger.debug(
                    f"[{request_id}] Batch {batch_num}/{total_embed_batches} embedded: "
                    f"{len(batch_with_embeddings)} docs in {time.time() - embed_start:.2f}s"
                )

            except Exception as e:
                stats["errors"] += len(batch_docs)
                stats["embedding_time"] += time.time() - embed_start

                logger.error(
                    f"[{request_id}] Embedding batch {batch_num} failed after retries: "
                    f"{type(e).__name__}: {e}"
                )

                if not self.config.continue_on_error:
                    raise EmbeddingError(
                        f"Batch {batch_num} embedding failed",
                        details={"batch_num": batch_num, "doc_count": len(batch_docs)},
                        original_error=e
                    )

            # Store when we have enough embedded documents
            if len(embedded_docs) >= store_batch_size:
                store_start = time.time()
                stored = self._store_batch(embedded_docs, stats, reporter, request_id)
                stats["stored"] += stored
                stats["storage_time"] += time.time() - store_start
                embedded_docs = []

        # Store remaining documents
        if embedded_docs:
            if reporter and self.config.show_progress:
                reporter.report(BatchProgress.create(
                    stage=BatchStage.STORING,
                    current=stats["stored"],
                    total=stats["embedded"],
                    message="Storing final batch",
                    errors=stats["errors"]
                ))

            store_start = time.time()
            stored = self._store_batch(embedded_docs, stats, reporter, request_id)
            stats["stored"] += stored
            stats["storage_time"] += time.time() - store_start

        # Report completion
        stats["duration"] = time.time() - start_time

        if reporter and self.config.show_progress:
            reporter.report(BatchProgress.create(
                stage=BatchStage.COMPLETE,
                current=total_docs,
                total=total_docs,
                message=f"Complete: {stats['stored']} stored, {stats['errors']} errors",
                errors=stats["errors"]
            ))

        # Log final summary
        logger.info(
            f"[{request_id}] Batch processing complete: "
            f"stored={stats['stored']}/{total_docs}, errors={stats['errors']}, "
            f"duration={stats['duration']:.2f}s "
            f"(embed={stats['embedding_time']:.2f}s, store={stats['storage_time']:.2f}s)"
        )

        return stats

    def _embed_batch(
        self,
        documents: list[Document],
        request_id: str,
        batch_num: int
    ) -> list[Document]:
        """
        Embed a batch of documents with retry logic.

        Uses exponential backoff with jitter for transient errors.
        Respects rate limit headers if present.

        Args:
            documents: Documents to embed
            request_id: Request ID for logging
            batch_num: Current batch number for logging

        Returns:
            Documents with embeddings attached

        Raises:
            Exception: If all retries fail
        """
        texts = [doc.page_content for doc in documents]

        @retry_with_backoff(config=self._retry_config)
        def _do_embed() -> list[list[float]]:
            return self.embedder.embed(texts)

        try:
            embeddings = _do_embed()

            # Attach embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.vector = embedding

            return documents

        except Exception as e:
            # Log detailed error info for debugging
            logger.error(
                f"[{request_id}] Embedding failed for batch {batch_num}: "
                f"error_type={type(e).__name__}, "
                f"retryable={is_retryable(e)}, "
                f"doc_count={len(documents)}"
            )
            raise

    def _store_batch(
        self,
        documents: list[Document],
        stats: dict[str, Any],
        reporter: CallbackProgressReporter | None,
        request_id: str
    ) -> int:
        """
        Store a batch of documents in the vector store.

        Args:
            documents: Documents to store
            stats: Statistics dict to update on error
            reporter: Progress reporter
            request_id: Request ID for logging

        Returns:
            Number of documents successfully stored
        """
        try:
            self.vector_store.add_texts(documents)

            logger.debug(
                f"[{request_id}] Stored {len(documents)} documents"
            )

            return len(documents)

        except Exception as e:
            logger.error(
                f"[{request_id}] Storage failed: {type(e).__name__}: {e}, "
                f"doc_count={len(documents)}"
            )

            stats["errors"] += len(documents)

            if not self.config.continue_on_error:
                raise IndexingError(
                    "Vector store insertion failed",
                    details={"doc_count": len(documents)},
                    original_error=e
                )

            return 0


def process_in_batches(
    embedder: BaseEmbedder,
    vector_store: BaseVector,
    documents: list[Document],
    config: BatchConfig | None = None,
    on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
    """
    Convenience function for batch processing documents.

    This is a simpler interface for one-off batch processing without
    needing to create a BatchProcessor instance.

    Args:
        embedder: Embedding model
        vector_store: Target vector store
        documents: Documents to process
        config: Optional batch configuration
        on_progress: Optional progress callback

    Returns:
        Processing statistics dictionary

    Example:
        >>> from langrag.batch import process_in_batches, BatchConfig
        >>> stats = process_in_batches(
        ...     embedder, store, docs,
        ...     config=BatchConfig(embedding_batch_size=50),
        ...     on_progress=lambda p: print(p.message)
        ... )
    """
    processor = BatchProcessor(embedder, vector_store, config)
    return processor.process_documents(documents, on_progress)
