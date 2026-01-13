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
        router = None
        rewriter = None
        reranker = None

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

        # Execute retrieval only - simplified version without complex workflow
        logger.info(f"[RAGChatService] Starting simplified retrieval with {len(kb_stores)} knowledge bases")

        # Apply rewriter if available
        original_query = query
        rewritten_query = None
        if rewriter:
            try:
                logger.info(f"[RAGChatService] Rewriting query: '{query}'")
                rewritten_query = rewriter.rewrite(query)
                if rewritten_query and rewritten_query != query:
                    logger.info(f"[RAGChatService] Query rewritten to: '{rewritten_query}'")
                    query = rewritten_query  # 使用重写后的 query 进行检索
                else:
                    logger.info(f"[RAGChatService] Query unchanged after rewrite")
                    rewritten_query = None
            except Exception as e:
                logger.warning(f"[RAGChatService] Rewriting failed: {e}")
                rewritten_query = None

        # Perform retrieval using vector stores directly
        import asyncio
        retrieval_results = []
        try:
            logger.info(f"[RAGChatService] Starting vector search with {len(kb_stores)} KBs")

            # Use the embedder passed to the service
            embedder = self.embedder
            if embedder:
                logger.info(f"[RAGChatService] Using configured embedder: {type(embedder)}")
            else:
                logger.warning(f"[RAGChatService] No embedder configured")

            # Embed the query
            query_vector = None
            if embedder:
                try:
                    logger.info(f"[RAGChatService] Embedding query: '{query[:50]}...'")
                    query_vectors = embedder.embed([query])
                    query_vector = query_vectors[0] if query_vectors else None
                    logger.info(f"[RAGChatService] Query embedded, vector length: {len(query_vector) if query_vector else 0}")
                except Exception as e:
                    logger.warning(f"[RAGChatService] Failed to embed query: {e}")
                    import traceback
                    logger.warning(f"[RAGChatService] Embedding traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[RAGChatService] No embedder available, trying search without vector")

            for kb_id, kb_store in kb_stores.items():
                try:
                    logger.info(f"[RAGChatService] Searching KB {kb_id} with store type: {type(kb_store)}")
                    # Use the vector store's search method
                    docs = kb_store.search(query, query_vector, top_k=top_k)
                    logger.info(f"[RAGChatService] KB {kb_id} returned {len(docs)} documents")
                    retrieval_results.extend(docs)
                except Exception as e:
                    logger.warning(f"[RAGChatService] Failed to search KB {kb_id}: {e}")

            logger.info(f"[RAGChatService] Total retrieval results: {len(retrieval_results)}")

            # Apply reranker if available
            if reranker and retrieval_results:
                try:
                    logger.info(f"[RAGChatService] Applying reranker to {len(retrieval_results)} documents")
                    retrieval_results = reranker.rerank(query, retrieval_results, top_k)
                    logger.info(f"[RAGChatService] Reranker returned {len(retrieval_results)} documents")
                except Exception as e:
                    logger.warning(f"[RAGChatService] Reranking failed: {e}")

        except Exception as e:
            logger.error(f"[RAGChatService] Retrieval failed: {e}")
            import traceback
            logger.error(f"[RAGChatService] Traceback: {traceback.format_exc()}")
            retrieval_results = []

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

        return {
            "query": original_query,
            "sources": sources,
            "rewritten_query": rewritten_query,
            "retrieval_stats": {
                "total_sources": len(sources),
                "knowledge_bases": len(kb_stores),
                "components_used": {
                    "router": router is not None,
                    "rewriter": rewriter is not None,
                    "reranker": reranker is not None
                }
            },
            "message": f"Retrieved {len(sources)} documents from {len(kb_stores)} knowledge bases"
        }