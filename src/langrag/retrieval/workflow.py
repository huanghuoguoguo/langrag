"""
Retrieval Workflow Module.

This module orchestrates the complete retrieval pipeline:
Query -> [Rewrite] -> [Route] -> [Retrieve] -> [Rerank] -> [PostProcess] -> Results

The workflow is broken into stages:
1. Query Rewriting - Optional query enhancement
2. Routing - Select relevant datasets
3. Retrieval - Fetch documents from selected datasets
4. Reranking - Reorder results by relevance
5. Post Processing - Deduplication and filtering
"""

import logging
import time
import uuid
from typing import Any

from langrag.entities.dataset import Dataset, RetrievalContext
from langrag.observability import get_tracer, is_tracing_enabled
from langrag.retrieval.config import WorkflowConfig, DEFAULT_MAX_WORKERS
from langrag.retrieval.executor import RetrievalExecutor
from langrag.retrieval.post_processor import PostProcessor
from langrag.utils.retry import RetryConfig, execute_with_retry


logger = logging.getLogger(__name__)


class RetrievalWorkflow:
    """
    Orchestrates the retrieval process:
    Query -> [Router] -> [RetrievalService] -> [Reranker] -> Results

    Features:
    - Parallel retrieval across multiple datasets
    - Graceful degradation (partial failures don't break entire retrieval)
    - Retry logic for transient errors
    - Comprehensive logging with request IDs
    - Timeout handling to prevent hung requests

    Example:
        workflow = RetrievalWorkflow(
            router=llm_router,
            reranker=reranker,
            config=WorkflowConfig(max_workers=10)
        )
        results = workflow.retrieve(query, datasets, top_k=5)
    """

    def __init__(
        self,
        router=None,   # BaseRouter
        embedder=None, # BaseEmbedder
        reranker=None, # BaseReranker
        rewriter=None, # BaseRewriter
        vector_store_cls=None,
        vector_manager=None, # Injected manager
        cache=None, # SemanticCache
        max_workers: int = DEFAULT_MAX_WORKERS,
        config: WorkflowConfig | None = None
    ):
        self.router = router
        self.embedder = embedder
        self.reranker = reranker
        self.rewriter = rewriter
        self.vector_store_cls = vector_store_cls
        self.vector_manager = vector_manager
        self.cache = cache
        self.config = config or WorkflowConfig(max_workers=max_workers)
        self.max_workers = self.config.max_workers
        self.post_processor = PostProcessor()
        self.callback_manager = None

        # Initialize retrieval executor
        self._executor = RetrievalExecutor(
            config=self.config,
            vector_store_cls=vector_store_cls
        )

    def set_callback_manager(self, manager):
        """Set the callback manager for lifecycle events."""
        self.callback_manager = manager

    def _embed_query(self, query: str) -> list[float] | None:
        """
        Generate embedding vector for the query.
        """
        if not self.embedder:
            return None

        try:
            vectors = self.embedder.embed([query])
            return vectors[0] if vectors else None
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def retrieve(
        self,
        query: str,
        datasets: list[Dataset],
        top_k: int = 4,
        score_threshold: float = 0.0,
        rerank_top_k: int | None = None
    ) -> list[RetrievalContext]:
        """
        Execute the retrieval workflow.

        Args:
            query: User query string.
            datasets: List of available datasets to potentially search.
            top_k: Initial retrieval count per dataset.
            score_threshold: Minimum score filter.
            rerank_top_k: How many results to return after reranking.

        Returns:
            List of RetrievalContext objects with retrieved content.

        Raises:
            RetrievalError: When retrieval fails completely (all datasets fail)
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        tracer = get_tracer()

        logger.info(
            f"[{request_id}] Retrieval started: "
            f"query_length={len(query)}, datasets={len(datasets)}, top_k={top_k}"
        )

        with tracer.start_as_current_span("retrieval_workflow") as workflow_span:
            if is_tracing_enabled():
                workflow_span.set_attribute("request_id", request_id)
                workflow_span.set_attribute("query", query)
                workflow_span.set_attribute("top_k", top_k)
                workflow_span.set_attribute("score_threshold", score_threshold)
                workflow_span.set_attribute("dataset_count", len(datasets))

            run_id = None
            if self.callback_manager:
                run_id = self.callback_manager.on_retrieve_start(query=query)

            try:
                # Stage 1: Query Rewrite
                final_query = self._stage_rewrite(query, request_id, tracer)
                
                # Check cache if enabled
                query_vector = None
                if self.cache and self.embedder:
                    query_vector = self._embed_query(final_query)
                    if query_vector:
                         context_key = ",".join(sorted([d.id for d in datasets])) if datasets else ""
                         
                         cache_hit = self.cache.get_by_similarity(query_vector, context_key=context_key)
                         
                         if cache_hit:
                             logger.info(f"[{request_id}] Cache hit for query: '{final_query[:30]}...'")
                             
                             # Convert cached docs to RetrievalContext
                             results = self._format_results(cache_hit.results, rerank_top_k or top_k)
                             
                             if self.callback_manager:
                                self.callback_manager.on_retrieve_end(results, run_id=run_id)
                                
                             return results

                # Stage 2: Routing
                selected_datasets = self._stage_route(final_query, datasets, request_id, tracer)

                if not selected_datasets:
                    logger.warning(f"[{request_id}] No datasets selected, returning empty")
                    if self.callback_manager:
                        self.callback_manager.on_retrieve_end([], run_id=run_id)
                    return []

                # Stage 3: Retrieval
                all_documents = self._stage_retrieve(
                    final_query, selected_datasets, top_k, request_id, tracer
                )

                logger.info(
                    f"[{request_id}] Retrieval completed: documents={len(all_documents)}"
                )

                # Stage 3.5: QA Processing
                self._process_qa_documents(all_documents)

                # Stage 4: Reranking
                if self.reranker and all_documents:
                    all_documents = self._stage_rerank(
                        final_query, all_documents, rerank_top_k or top_k,
                        request_id, run_id, tracer
                    )

                # Stage 5: Post Processing
                all_documents = self._stage_post_process(
                    all_documents, score_threshold, tracer
                )
                
                # Update Cache if applicable
                if self.cache and query_vector and all_documents:
                    context_key = ",".join(sorted([d.id for d in datasets])) if datasets else ""
                    self.cache.set_with_embedding(
                        query=final_query,
                        embedding=query_vector,
                        results=all_documents,
                        metadata={
                            "top_k": top_k,
                            "context_key": context_key
                        }
                    )

                # Stage 6: Format Results
                results = self._format_results(all_documents, rerank_top_k or top_k)

                elapsed = time.time() - start_time
                logger.info(
                    f"[{request_id}] Workflow completed: "
                    f"results={len(results)}, elapsed={elapsed:.2f}s"
                )

                if is_tracing_enabled():
                    workflow_span.set_attribute("result_count", len(results))
                    workflow_span.set_attribute("elapsed_seconds", elapsed)

                if self.callback_manager:
                    self.callback_manager.on_retrieve_end(results, run_id=run_id)

                return results

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"[{request_id}] Workflow failed after {elapsed:.2f}s: "
                    f"{type(e).__name__}: {e}"
                )

                if is_tracing_enabled():
                    workflow_span.record_exception(e)
                    workflow_span.set_attribute("error", str(e))

                if self.callback_manager:
                    self.callback_manager.on_error(e, run_id=run_id)

                raise

    # -------------------------------------------------------------------------
    # Pipeline Stages
    # -------------------------------------------------------------------------

    def _stage_rewrite(self, query: str, request_id: str, tracer: Any) -> str:
        """
        Stage 1: Query Rewriting.

        Falls back to original query on failure.
        """
        if not self.rewriter:
            return query

        original_query = query

        with tracer.start_as_current_span("query_rewrite") as span:
            try:
                rewritten = self.rewriter.rewrite(query)

                if is_tracing_enabled():
                    span.set_attribute("original_query", query)
                    span.set_attribute("rewritten_query", rewritten)

                if rewritten != original_query:
                    logger.info(
                        f"[{request_id}] Query rewritten: "
                        f"'{original_query[:50]}...' -> '{rewritten[:50]}...'"
                    )

                return rewritten

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Query rewrite failed, using original: "
                    f"{type(e).__name__}: {e}"
                )

                if is_tracing_enabled():
                    span.record_exception(e)
                    span.set_attribute("fallback", True)

                return query

    def _stage_route(
        self,
        query: str,
        datasets: list[Dataset],
        request_id: str,
        tracer: Any
    ) -> list[Dataset]:
        """
        Stage 2: Dataset Routing.

        Falls back to all datasets on routing failure.
        """
        if not self.router or len(datasets) <= 1:
            return datasets

        with tracer.start_as_current_span("routing") as span:
            try:
                if self.config.enable_router_retry:
                    retry_config = RetryConfig(
                        max_attempts=self.config.router_max_retries,
                        base_delay=0.5,
                        max_delay=5.0,
                    )
                    selected = execute_with_retry(
                        self.router.route,
                        query,
                        datasets,
                        config=retry_config
                    )
                else:
                    selected = self.router.route(query, datasets)

                logger.info(
                    f"[{request_id}] Router selected {len(selected)}/{len(datasets)} datasets: "
                    f"{[d.name for d in selected]}"
                )

                if is_tracing_enabled():
                    span.set_attribute("selected_count", len(selected))
                    span.set_attribute("selected_datasets", [d.name for d in selected])

                return selected # Trust the router even if empty

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Router failed, falling back to all datasets: "
                    f"{type(e).__name__}: {e}"
                )

                if is_tracing_enabled():
                    span.record_exception(e)
                    span.set_attribute("fallback", True)

                return datasets

    def _stage_retrieve(
        self,
        query: str,
        datasets: list[Dataset],
        top_k: int,
        request_id: str,
        tracer: Any
    ) -> list:
        """
        Stage 3: Document Retrieval.

        Delegates to RetrievalExecutor for parallel/single retrieval.
        """
        with tracer.start_as_current_span("vector_retrieval") as span:
            if is_tracing_enabled():
                span.set_attribute("dataset_count", len(datasets))

            documents = self._executor.execute(
                query, datasets, top_k, request_id, span, self.vector_manager
            )

            if is_tracing_enabled():
                span.set_attribute("retrieved_count", len(documents))

            return documents

    def _stage_rerank(
        self,
        query: str,
        documents: list,
        top_k: int,
        request_id: str,
        run_id: str | None,
        tracer: Any
    ) -> list:
        """
        Stage 4: Document Reranking.

        Falls back to original order on reranker failure.
        """
        with tracer.start_as_current_span("reranking") as span:
            if is_tracing_enabled():
                span.set_attribute("input_count", len(documents))

            if self.callback_manager:
                self.callback_manager.on_rerank_start(
                    query=query, documents=documents, run_id=run_id
                )

            try:
                from langrag.entities.search_result import SearchResult

                search_results = [
                    SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
                    for doc in documents
                ]

                reranked_results = self.reranker.rerank(
                    query, search_results, top_k=top_k
                )

                result_documents = []
                for res in reranked_results:
                    doc = res.chunk
                    doc.metadata['score'] = res.score
                    result_documents.append(doc)

                logger.debug(
                    f"[{request_id}] Reranking completed: "
                    f"{len(documents)} -> {len(result_documents)}"
                )

                if is_tracing_enabled():
                    span.set_attribute("output_count", len(result_documents))

                if self.callback_manager:
                    self.callback_manager.on_rerank_end(
                        documents=result_documents, run_id=run_id
                    )

                return result_documents

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Reranking failed, using original order: "
                    f"{type(e).__name__}: {e}"
                )

                if is_tracing_enabled():
                    span.record_exception(e)
                    span.set_attribute("fallback", True)

                if self.callback_manager:
                    self.callback_manager.on_error(e, run_id=run_id)
                    self.callback_manager.on_rerank_end(documents=documents, run_id=run_id)

                return documents

    def _stage_post_process(
        self,
        documents: list,
        score_threshold: float,
        tracer: Any
    ) -> list:
        """
        Stage 5: Post Processing (deduplication and filtering).
        """
        with tracer.start_as_current_span("post_processing") as span:
            if is_tracing_enabled():
                span.set_attribute("input_count", len(documents))

            processed = self.post_processor.run(
                documents, score_threshold=score_threshold
            )

            if is_tracing_enabled():
                span.set_attribute("output_count", len(processed))

            return processed

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _process_qa_documents(self, documents: list) -> None:
        """
        Process QA-indexed documents by swapping question content with answer.
        """
        for doc in documents:
            if doc.metadata.get("is_qa"):
                question_text = doc.page_content
                doc.metadata["matched_question"] = question_text

                if "answer" in doc.metadata:
                    doc.page_content = doc.metadata["answer"]

                if "original_doc_id" in doc.metadata:
                    doc.id = doc.metadata["original_doc_id"]
                    doc.metadata["document_id"] = doc.metadata["original_doc_id"]

    def _format_results(
        self,
        documents: list,
        top_k: int
    ) -> list[RetrievalContext]:
        """
        Format documents into RetrievalContext objects.
        """
        results = []
        for doc in documents:
            score = doc.metadata.get('score', 0.0)

            results.append(RetrievalContext(
                document_id=doc.metadata.get('document_id', 'unknown'),
                content=doc.page_content,
                score=score,
                metadata=doc.metadata
            ))

        # Sort if no reranker was used
        if not self.reranker:
            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]
