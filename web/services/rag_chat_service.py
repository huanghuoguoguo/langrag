"""
RAG Chat Service for the Web layer.

This service handles RAG retrieval with advanced configuration,
returning relevant documents for external LLM processing.
"""

import logging
from typing import Dict, List, Any

from langrag import BaseVector
from langrag.entities.dataset import RetrievalContext
from langrag.llm.base import ModelManager
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.retrieval.router.base import BaseRouter
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.retrieval.rerank.factory import RerankerFactory
from langrag.retrieval.rerank.providers.llm_template import LLMTemplateReranker

logger = logging.getLogger(__name__)


class RAGChatService:
    """
    RAG Chat Service that performs retrieval with dynamic component configuration.

    This service retrieves relevant documents using configurable RAG components
    (router, rewriter, reranker) and returns them for external LLM processing.
    It does NOT perform final answer generation - that's left to external services.
    """

    def __init__(self, model_manager: ModelManager, embedder=None):
        """
        Initialize the RAG chat service.

        Args:
            model_manager: ModelManager instance for accessing configured LLMs
            embedder: Optional embedder instance for query vectorization
        """
        self.model_manager = model_manager
        self.embedder = embedder

    async def retrieve(
        self,
        kb_ids: list[str],
        kb_stores: Dict[str, BaseVector],
        query: str,
        # Retrieval configuration
        use_rerank: bool = False,
        reranker_type: str | None = None,
        reranker_model: str | None = None,
        use_router: bool = False,
        router_model: str | None = None,
        use_rewriter: bool = False,
        rewriter_model: str | None = None,
        # Additional config
        top_k: int = 5,
        kb_names: Dict[str, str] | None = None  # 知识库ID到名称的映射
    ) -> Dict[str, Any]:
        """
        Perform RAG retrieval with dynamic component configuration.

        This method only retrieves relevant documents - it does NOT generate
        final answers. The retrieved documents should be passed to an external
        LLM service for answer generation.

        Args:
            kb_ids: List of knowledge base IDs
            kb_stores: Dictionary mapping kb_id to BaseVector stores
            query: User query
            use_rerank: Whether to enable reranking
            reranker_type: Type of reranker ('llm_template', 'cohere', 'qwen', 'noop')
            reranker_model: Model for LLM template reranker
            use_router: Whether to enable routing
            router_model: Model for router
            use_rewriter: Whether to enable query rewriting
            rewriter_model: Model for rewriter
            top_k: Number of documents to retrieve
            kb_names: Optional dictionary mapping kb_id to kb_name for display purposes

        Returns:
            Dictionary containing:
            - sources: List of retrieved documents with metadata
            - rewritten_query: Rewritten query if rewriting was applied, None otherwise
            - retrieval_stats: Statistics about the retrieval process
        """
        # Determine components for this request
        from langrag.observability import get_tracer, is_tracing_enabled
        tracer = get_tracer()

        router = None
        rewriter = None
        reranker = None

        with tracer.start_as_current_span("rag_service_retrieve") as workflow_span:
            if is_tracing_enabled():
                workflow_span.set_attribute("kb_count", len(kb_stores))
                workflow_span.set_attribute("query", query)
                workflow_span.set_attribute("use_rerank", use_rerank)
                workflow_span.set_attribute("use_router", use_router)
                workflow_span.set_attribute("use_rewriter", use_rewriter)

            # Debug: Log input parameters
            logger.info(f"[RAGChatService] Input params: use_router={use_router}, router_model={router_model}, use_rewriter={use_rewriter}, rewriter_model={rewriter_model}, use_rerank={use_rerank}, reranker_type={reranker_type}, reranker_model={reranker_model}")

            # Get router if enabled
            if use_router and router_model:
                logger.info(f"[RAGChatService] Creating router with model '{router_model}'")
                router_llm = self.model_manager.get_model(router_model)
                if router_llm:
                    from langrag.retrieval.router.llm_router import LLMRouter
                    router = LLMRouter(llm=router_llm)
                    logger.info(f"[RAGChatService] Router enabled with model '{router_model}'")
                else:
                    logger.warning(f"[RAGChatService] Router model '{router_model}' not found")

            # Get rewriter if enabled
            if use_rewriter and rewriter_model:
                logger.info(f"[RAGChatService] Creating rewriter with model '{rewriter_model}'")
                rewriter_llm = self.model_manager.get_model(rewriter_model)
                if rewriter_llm:
                    from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
                    rewriter = LLMRewriter(llm=rewriter_llm)
                    logger.info(f"[RAGChatService] Rewriter enabled with model '{rewriter_model}'")
                else:
                    logger.warning(f"[RAGChatService] Rewriter model '{rewriter_model}' not found")

            # Get reranker if enabled
            if use_rerank and reranker_type:
                try:
                    if reranker_type == "llm_template":
                        # For LLM template reranker, we need an LLM model
                        reranker_llm = None
                        if reranker_model:
                            reranker_llm = self.model_manager.get_model(reranker_model)

                        if reranker_llm:
                            reranker = LLMTemplateReranker(llm_model=reranker_llm)
                            logger.info(f"[RAGChatService] LLM Template reranker enabled with model '{getattr(reranker_llm, 'model', 'unknown')}'")
                        else:
                            logger.warning("[RAGChatService] No LLM available for LLM template reranker")
                    else:
                        reranker = RerankerFactory.create(reranker_type)
                        logger.info(f"[RAGChatService] Reranker enabled: {reranker_type}")
                except Exception as e:
                    logger.warning(f"[RAGChatService] Failed to create reranker '{reranker_type}': {e}")
                    if is_tracing_enabled():
                         workflow_span.record_exception(e)

            # Execute retrieval only - simplified version without complex workflow
            logger.info(f"[RAGChatService] Starting simplified retrieval with {len(kb_stores)} knowledge bases")

            # Apply rewriter if available
            original_query = query
            rewritten_query = None
            if rewriter:
                with tracer.start_as_current_span("rewrite_query") as rewrite_span:
                    try:
                        logger.info(f"[RAGChatService] Rewriting query: '{query}'")
                        rewritten_query = rewriter.rewrite(query)
                        if rewritten_query and rewritten_query != query:
                            logger.info(f"[RAGChatService] Query rewritten to: '{rewritten_query}'")
                            query = rewritten_query  # 使用重写后的 query 进行检索
                            if is_tracing_enabled():
                                rewrite_span.set_attribute("original_query", original_query)
                                rewrite_span.set_attribute("rewritten_query", rewritten_query)
                        else:
                            logger.info(f"[RAGChatService] Query unchanged after rewrite")
                            rewritten_query = None
                    except Exception as e:
                        logger.warning(f"[RAGChatService] Rewriting failed: {e}")
                        rewritten_query = None
                        if is_tracing_enabled():
                            rewrite_span.record_exception(e)

            # Perform retrieval using vector stores directly
            import asyncio
            retrieval_results = []
            
            # Router Logic
            target_kb_stores = kb_stores
            selected_kbs_info = []

            # If router is enabled, determine which KBs to search
            # We use 'rewritten_query' if available, otherwise 'query' for routing decision
            routing_query = query 
            
            if router:
                with tracer.start_as_current_span("router_selection") as router_span:
                    try:
                        from langrag.entities.dataset import Dataset
                        
                        # Construct options for router from available stores
                        # Use name if available, else ID
                        route_options = []
                        for kbid in kb_stores.keys():
                            kb_name = kb_names.get(kbid, kbid) if kb_names else kbid
                            # Simple description for now
                            route_options.append(Dataset(id=kbid, name=kb_name, description=f"Knowledge Base: {kb_name}"))
                        
                        logger.info(f"[RAGChatService] Routing query '{routing_query}' to {len(route_options)} candidate KBs")
                        
                        # Call router (synchronous currently)
                        selected_datasets = router.route(routing_query, route_options)
                        selected_kb_ids = [d.name for d in selected_datasets] # Note: LLMRouter returns Datasets with names populated from user selection
                        
                        # Careful: LLMRouter returns new Dataset objects based on names
                        # We need to match names back to IDs if names were used for routing
                        # But here we constructed Datasets with name=kb_name
                        
                        # Let's map back.
                        # target_kb_stores = {k: v for k, v in kb_stores.items() if (kb_names.get(k, k) in selected_kb_ids)}
                        
                        # Actually LLMRouter implementation returns the subset of the original Dataset list we passed!
                        # See standard implementation: return [d for d in datasets if d.name in names]
                        # So we can trust the IDs in the returned datasets.
                        
                        final_ids = [d.id for d in selected_datasets]

                        target_kb_stores = {
                            k: v for k, v in kb_stores.items() 
                            if k in final_ids
                        }
                        
                        logger.info(f"[RAGChatService] Router selected {len(target_kb_stores)} KBs: {list(target_kb_stores.keys())}")
                        
                        selected_kbs_info = [{"id": k, "name": kb_names.get(k, k) if kb_names else k} for k in target_kb_stores.keys()]
                        
                        if is_tracing_enabled():
                            router_span.set_attribute("selected_count", len(target_kb_stores))
                            
                    except Exception as e:
                        logger.error(f"[RAGChatService] Routing failed: {e}")
                        # Fallback to all
                        target_kb_stores = kb_stores 
                        if is_tracing_enabled():
                            router_span.record_exception(e)

            try:
                with tracer.start_as_current_span("vector_search") as search_span:
                    logger.info(f"[RAGChatService] Starting vector search with {len(target_kb_stores)} KBs")

                    # Use the embedder passed to the service
                    embedder = self.embedder
                    if embedder:
                        logger.info(f"[RAGChatService] Using configured embedder: {type(embedder)}")
                    else:
                        logger.warning(f"[RAGChatService] No embedder configured")

                    # Embed the query (using the potentially rewritten query)
                    query_vector = None
                    if embedder:
                        try:
                            # Use current effective query
                            logger.info(f"[RAGChatService] Embedding query: '{query[:50]}...'")
                            query_vectors = embedder.embed([query])
                            query_vector = query_vectors[0] if query_vectors else None
                            logger.info(f"[RAGChatService] Query embedded, vector length: {len(query_vector) if query_vector else 0}")
                        except Exception as e:
                            logger.warning(f"[RAGChatService] Failed to embed query: {e}")
                            import traceback
                            logger.warning(f"[RAGChatService] Embedding traceback: {traceback.format_exc()}")
                            if is_tracing_enabled():
                                search_span.record_exception(e)
                    else:
                        logger.warning(f"[RAGChatService] No embedder available, trying search without vector")

                    for kb_id, kb_store in target_kb_stores.items():
                        try:
                            logger.info(f"[RAGChatService] Searching KB {kb_id} with store type: {type(kb_store)}")
                            # Use the vector store's search method
                            docs = kb_store.search(query, query_vector, top_k=top_k)
                            logger.info(f"[RAGChatService] KB {kb_id} returned {len(docs)} documents")
                            retrieval_results.extend(docs)
                        except Exception as e:
                            logger.warning(f"[RAGChatService] Failed to search KB {kb_id}: {e}")
                            if is_tracing_enabled():
                                search_span.record_exception(e)

                    logger.info(f"[RAGChatService] Total retrieval results: {len(retrieval_results)}")
                    if is_tracing_enabled():
                        search_span.set_attribute("total_results", len(retrieval_results))

                # Apply reranker if available
                rerank_input_count = len(retrieval_results)
                rerank_output_count = 0
                
                if reranker and retrieval_results:
                    with tracer.start_as_current_span("rerank_results") as rerank_span:
                        try:
                            logger.info(f"[RAGChatService] Applying reranker to {len(retrieval_results)} documents")
                            retrieval_results = reranker.rerank(query, retrieval_results, top_k)
                            rerank_output_count = len(retrieval_results)
                            logger.info(f"[RAGChatService] Reranker returned {len(retrieval_results)} documents")
                            if is_tracing_enabled():
                                rerank_span.set_attribute("input_count", rerank_input_count)
                                rerank_span.set_attribute("output_count", rerank_output_count)
                        except Exception as e:
                            logger.warning(f"[RAGChatService] Reranking failed: {e}")
                            if is_tracing_enabled():
                                rerank_span.record_exception(e)
                else:
                    rerank_output_count = len(retrieval_results)

            except Exception as e:
                logger.error(f"[RAGChatService] Retrieval failed: {e}")
                import traceback
                logger.error(f"[RAGChatService] Traceback: {traceback.format_exc()}")
                retrieval_results = []
                if is_tracing_enabled():
                    workflow_span.record_exception(e)

            # Format results
            sources = []
            try:
                for result in retrieval_results:
                    # Find which KB this result came from
                    metadata = getattr(result, 'metadata', {}) or {}
                    kb_id = metadata.get('kb_id', 'unknown')
                    kb_name = kb_names.get(kb_id, kb_id) if kb_names else kb_id

                    sources.append({
                        "content": getattr(result, 'content', ''),
                        "score": getattr(result, 'score', 0.0),
                        "source": metadata.get('source', 'unknown'),
                        "kb_id": kb_id,
                        "kb_name": kb_name,
                        "document_id": getattr(result, 'document_id', ''),
                        "metadata": metadata
                    })
            except Exception as e:
                logger.error(f"[RAGChatService] Error formatting results: {e}")
                sources = []

                # Format simplified selected KBs list for stats
            # Format simplified selected KBs list for stats
            # Use 'selected_kbs_info' from router logic if available, otherwise build from target_kb_stores
            if not selected_kbs_info and 'target_kb_stores' in locals():
                selected_kbs_info = [{"id": kid, "name": kb_names.get(kid, kid) if kb_names else kid} for kid in target_kb_stores.keys()]
            elif not selected_kbs_info and 'kb_stores' in locals():
                 selected_kbs_info = [{"id": kid, "name": kb_names.get(kid, kid) if kb_names else kid} for kid in kb_stores.keys()]

            return {
                "query": original_query,
                "sources": sources,
                "rewritten_query": rewritten_query,
                "retrieval_stats": {
                    "source_count": len(sources),
                    "kb_count": len(target_kb_stores) if 'target_kb_stores' in locals() else len(kb_stores),
                    "pipeline": {
                        "rewriter": {
                            "enabled": bool(rewriter),
                            "model": getattr(rewriter.llm, "model", "unknown") if hasattr(rewriter, "llm") else getattr(rewriter, "model_name", "unknown"),
                            "original": original_query,
                            "output": query if query and query != original_query else original_query
                        },
                        "router": {
                            "enabled": bool(router),
                            "model": getattr(router.llm, "model", "unknown") if hasattr(router, "llm") else "unknown",
                            "selected_kbs": selected_kbs_info,
                            "all_kbs_count": len(kb_names) if kb_names else 0
                        },
                        "reranker": {
                            "enabled": bool(reranker),
                            "model": getattr(reranker.llm_model, "model", "unknown") if hasattr(reranker, "llm_model") else getattr(reranker, "model_name", "unknown"),
                            "input_count": rerank_input_count,
                            "output_count": rerank_output_count
                        }
                    }
                },
                "message": f"Retrieved {len(sources)} documents"
            }