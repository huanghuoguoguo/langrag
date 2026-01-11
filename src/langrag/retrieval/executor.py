"""
Retrieval Executor Module.

Handles parallel and single dataset retrieval with retry and timeout support.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any

from langrag.datasource.service import RetrievalService
from langrag.entities.dataset import Dataset
from langrag.observability import is_tracing_enabled
from langrag.retrieval.config import WorkflowConfig
from langrag.utils.retry import RetryConfig, execute_with_retry


logger = logging.getLogger(__name__)


class RetrievalExecutor:
    """
    Executes retrieval operations against datasets.

    Supports:
    - Parallel retrieval across multiple datasets
    - Single dataset retrieval
    - Retry logic for transient failures
    - Timeout handling
    - Graceful degradation on partial failures
    """

    def __init__(
        self,
        config: WorkflowConfig,
        vector_store_cls: Any = None
    ):
        """
        Initialize the executor.

        Args:
            config: Workflow configuration
            vector_store_cls: Optional vector store class override
        """
        self.config = config
        self.vector_store_cls = vector_store_cls

    def execute(
        self,
        query: str,
        datasets: list[Dataset],
        top_k: int,
        request_id: str,
        tracer_span: Any
    ) -> list:
        """
        Execute retrieval from datasets.

        Args:
            query: Search query
            datasets: List of datasets to search
            top_k: Number of results per dataset
            request_id: Request ID for logging
            tracer_span: OpenTelemetry span for tracing

        Returns:
            List of retrieved documents
        """
        if len(datasets) > 1:
            return self._retrieve_parallel(
                query, datasets, top_k, request_id, tracer_span
            )
        return self._retrieve_single(
            query, datasets[0], top_k, request_id, tracer_span
        )

    def _get_retrieval_method(self, dataset: Dataset) -> str:
        """
        Determine retrieval method based on dataset configuration.

        Args:
            dataset: Dataset to check

        Returns:
            Retrieval method: 'keyword_search' or 'semantic_search'
        """
        if dataset.indexing_technique == 'economy':
            return "keyword_search"
        return "semantic_search"

    def _retrieve_single(
        self,
        query: str,
        dataset: Dataset,
        top_k: int,
        request_id: str,
        tracer_span: Any
    ) -> list:
        """
        Retrieve documents from a single dataset.

        Args:
            query: Search query
            dataset: Dataset to search
            top_k: Number of results
            request_id: Request ID for logging
            tracer_span: OpenTelemetry span

        Returns:
            List of documents (empty on failure)
        """
        start_time = time.time()

        try:
            method = self._get_retrieval_method(dataset)

            if self.config.enable_retrieval_retry:
                retry_config = RetryConfig(
                    max_attempts=self.config.retrieval_max_retries,
                    base_delay=0.5,
                    max_delay=5.0,
                )
                docs = execute_with_retry(
                    RetrievalService.retrieve,
                    dataset=dataset,
                    query=query,
                    retrieval_method=method,
                    top_k=top_k,
                    vector_store_cls=self.vector_store_cls,
                    config=retry_config
                )
            else:
                docs = RetrievalService.retrieve(
                    dataset=dataset,
                    query=query,
                    retrieval_method=method,
                    top_k=top_k,
                    vector_store_cls=self.vector_store_cls
                )

            elapsed = time.time() - start_time
            logger.debug(
                f"[{request_id}] Retrieved {len(docs)} docs from {dataset.name} "
                f"in {elapsed:.2f}s"
            )

            return docs

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] Retrieval from {dataset.name} failed after {elapsed:.2f}s: "
                f"{type(e).__name__}: {e}"
            )

            if is_tracing_enabled():
                tracer_span.record_exception(e)

            return []

    def _retrieve_parallel(
        self,
        query: str,
        datasets: list[Dataset],
        top_k: int,
        request_id: str,
        tracer_span: Any
    ) -> list:
        """
        Retrieve documents from multiple datasets in parallel.

        Uses ThreadPoolExecutor with timeout to prevent hung retrievals.
        Partial failures are logged but don't prevent successful results.

        Args:
            query: Search query
            datasets: List of datasets to search
            top_k: Number of results per dataset
            request_id: Request ID for logging
            tracer_span: OpenTelemetry span

        Returns:
            Combined list of documents from all successful retrievals
        """
        all_documents = []
        errors = []
        completed_count = 0
        timed_out_count = 0

        num_workers = min(len(datasets), self.config.max_workers)

        def retrieve_from_dataset(dataset: Dataset) -> tuple[Dataset, list, Exception | None]:
            """Worker function for parallel retrieval."""
            try:
                method = self._get_retrieval_method(dataset)

                if self.config.enable_retrieval_retry:
                    retry_config = RetryConfig(
                        max_attempts=self.config.retrieval_max_retries,
                        base_delay=0.5,
                        max_delay=5.0,
                    )
                    docs = execute_with_retry(
                        RetrievalService.retrieve,
                        dataset=dataset,
                        query=query,
                        retrieval_method=method,
                        top_k=top_k,
                        vector_store_cls=self.vector_store_cls,
                        config=retry_config
                    )
                else:
                    docs = RetrievalService.retrieve(
                        dataset=dataset,
                        query=query,
                        retrieval_method=method,
                        top_k=top_k,
                        vector_store_cls=self.vector_store_cls
                    )

                return (dataset, docs, None)
            except Exception as e:
                return (dataset, [], e)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_dataset = {
                executor.submit(retrieve_from_dataset, ds): ds
                for ds in datasets
            }

            try:
                for future in as_completed(
                    future_to_dataset,
                    timeout=self.config.retrieval_timeout
                ):
                    dataset, docs, error = future.result()

                    if error:
                        errors.append((dataset.name, error))
                        logger.warning(
                            f"[{request_id}] Retrieval from {dataset.name} failed: "
                            f"{type(error).__name__}: {error}"
                        )

                        if is_tracing_enabled():
                            tracer_span.record_exception(error)
                    else:
                        completed_count += 1
                        all_documents.extend(docs)
                        logger.debug(
                            f"[{request_id}] Retrieved {len(docs)} from {dataset.name}"
                        )

            except FuturesTimeoutError:
                timed_out_count = len(datasets) - completed_count - len(errors)
                logger.warning(
                    f"[{request_id}] Parallel retrieval timed out after "
                    f"{self.config.retrieval_timeout}s. "
                    f"Completed: {completed_count}, Timed out: {timed_out_count}"
                )

        elapsed = time.time() - start_time

        # Log summary
        if errors or timed_out_count > 0:
            logger.warning(
                f"[{request_id}] Parallel retrieval completed with issues: "
                f"successful={completed_count}/{len(datasets)}, "
                f"errors={len(errors)}, timed_out={timed_out_count}, "
                f"documents={len(all_documents)}, elapsed={elapsed:.2f}s"
            )
        else:
            logger.info(
                f"[{request_id}] Parallel retrieval successful: "
                f"documents={len(all_documents)} from {len(datasets)} datasets "
                f"in {elapsed:.2f}s"
            )

        if is_tracing_enabled():
            tracer_span.set_attribute("parallel_workers", num_workers)
            tracer_span.set_attribute("error_count", len(errors))
            tracer_span.set_attribute("timeout_count", timed_out_count)
            tracer_span.set_attribute("completed_count", completed_count)

        return all_documents
