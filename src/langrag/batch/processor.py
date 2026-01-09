"""
Batch processor for efficient document processing.

This module provides the main BatchProcessor class that handles
efficient processing of large document collections with:
- Configurable batch sizes for embedding and storage
- Progress tracking via callbacks
- Error handling with optional retry logic
- Memory-efficient chunked processing
"""

import logging
import time
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
from langrag.llm.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Efficient batch processor for document embedding and storage.

    This processor handles the embedding and storage of large document
    collections by processing them in configurable batches, with support
    for progress tracking and error handling.

    The processing flow:
    1. Split documents into embedding batches
    2. Generate embeddings for each batch
    3. Accumulate embedded documents
    4. Store in storage batches when threshold reached
    5. Report progress throughout

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

        Raises:
            Exception: If continue_on_error is False and an error occurs
        """
        start_time = time.time()
        stats = {
            "total": len(documents),
            "embedded": 0,
            "stored": 0,
            "errors": 0,
            "duration": 0.0
        }

        if not documents:
            return stats

        reporter = CallbackProgressReporter(on_progress) if on_progress else None
        embedded_docs: list[Document] = []
        total_docs = len(documents)

        # Calculate batch counts
        embed_batch_size = self.config.embedding_batch_size
        store_batch_size = self.config.storage_batch_size
        total_embed_batches = (total_docs + embed_batch_size - 1) // embed_batch_size

        logger.info(
            f"Starting batch processing: {total_docs} documents, "
            f"embed_batch={embed_batch_size}, store_batch={store_batch_size}"
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
            try:
                batch_with_embeddings = self._embed_batch(batch_docs)
                embedded_docs.extend(batch_with_embeddings)
                stats["embedded"] += len(batch_with_embeddings)
            except Exception as e:
                stats["errors"] += len(batch_docs)
                logger.error(f"Embedding batch {batch_num} failed: {e}")
                if not self.config.continue_on_error:
                    raise

            # Store when we have enough embedded documents
            if len(embedded_docs) >= store_batch_size:
                stored = self._store_batch(embedded_docs, stats, reporter)
                stats["stored"] += stored
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
            stored = self._store_batch(embedded_docs, stats, reporter)
            stats["stored"] += stored

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

        logger.info(
            f"Batch processing complete: {stats['stored']}/{total_docs} stored, "
            f"{stats['errors']} errors, {stats['duration']:.2f}s"
        )

        return stats

    def _embed_batch(self, documents: list[Document]) -> list[Document]:
        """
        Embed a batch of documents with retry logic.

        Args:
            documents: Documents to embed

        Returns:
            Documents with embeddings attached

        Raises:
            Exception: If all retries fail
        """
        texts = [doc.page_content for doc in documents]
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                embeddings = self.embedder.embed(texts)

                # Attach embeddings to documents
                for doc, embedding in zip(documents, embeddings):
                    doc.vector = embedding

                return documents

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Embedding attempt {attempt + 1} failed, "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)

        raise last_error

    def _store_batch(
        self,
        documents: list[Document],
        stats: dict[str, Any],
        reporter: CallbackProgressReporter | None
    ) -> int:
        """
        Store a batch of documents in the vector store.

        Args:
            documents: Documents to store
            stats: Statistics dict to update on error
            reporter: Progress reporter

        Returns:
            Number of documents successfully stored
        """
        try:
            self.vector_store.add_texts(documents)
            return len(documents)

        except Exception as e:
            logger.error(f"Storage batch failed: {e}")
            stats["errors"] += len(documents)
            if not self.config.continue_on_error:
                raise
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
