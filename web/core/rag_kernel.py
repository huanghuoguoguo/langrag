"""
RAG Kernel - Simplified coordinator for LangRAG Web Application.

This is a minimal implementation that:
1. Supports stage-based model configuration (different LLM for different tasks)
2. Delegates RAG operations to langrag high-level API
3. Manages vector stores and multiple embedders
4. Integrates with WebModelManager for LLM instance management
"""

import logging
from pathlib import Path
from typing import Any

import langrag
from langrag import BaseEmbedder, BaseVector, Dataset, EmbedderFactory
from langrag import Document as LangRAGDocument
from langrag.cache import SemanticCache
from langrag.datasource.kv.sqlite import SQLiteKV
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rerank.factory import RerankerFactory
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
from langrag.retrieval.router.llm_router import LLMRouter
from langrag.llm.base import BaseLLM

from web.config import DATA_DIR, settings
from web.core.vdb_manager import WebVectorStoreManager
from web.core.embedder_manager import EmbedderManager
from web.core.embedders import WebOpenAIEmbedder
from web.core.model_manager import WebModelManager

logger = logging.getLogger(__name__)


class RAGKernel:
    """
    Simplified RAG Kernel with stage-based model configuration.

    Different RAG stages can use different models:
    - chat: Main conversation/generation (use powerful model)
    - rewrite: Query rewriting (can use cheaper model)
    - router: KB routing (can use cheaper model)
    - rerank: Result reranking (use reranker model)
    - qa_indexing: QA pair generation (can use cheaper model)

    Example:
        kernel = RAGKernel()

        # Set different models for different stages
        kernel.set_stage_model("chat", gpt4_llm)
        kernel.set_stage_model("rewrite", gpt35_llm)
        kernel.set_stage_model("rerank", cohere_reranker)

        # Search uses configured models automatically
        results = kernel.search(kb_id, query)
    """

    # Available stages
    STAGES = ["chat", "rewrite", "router", "rerank", "qa_indexing"]

    def __init__(self):
        # Model manager for LLM instance management
        self.model_manager = WebModelManager()

        # Embedder manager for multiple embedders (each KB can have its own)
        self.embedder_manager = EmbedderManager()

        # Stage -> Model mapping (for non-LLM models like rerankers)
        self._stage_models: dict[str, BaseLLM | BaseReranker | BaseRewriter | None] = {
            stage: None for stage in self.STAGES
        }

        # Default LLM
        self.default_llm: BaseLLM | None = None

        # Vector stores
        self.vdb_manager = WebVectorStoreManager()
        self._stores: dict[str, BaseVector] = {}

        # KV store for parent-child indexing
        self.kv_store = SQLiteKV(db_path=str(DATA_DIR / "kv_store.sqlite"))

        # Semantic cache
        self.cache: SemanticCache | None = None
        if settings.CACHE_ENABLED:
            self.cache = SemanticCache(
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
                ttl_seconds=settings.CACHE_TTL_SECONDS,
                max_size=settings.CACHE_MAX_SIZE
            )

        # Register custom embedder
        try:
            EmbedderFactory.register("web_openai", WebOpenAIEmbedder)
        except Exception:
            pass

        logger.info("[RAGKernel] Initialized")

    # Backward compatibility: single embedder property
    @property
    def embedder(self) -> BaseEmbedder | None:
        """Get the default embedder (for backward compatibility)."""
        return self.embedder_manager.get()

    @embedder.setter
    def embedder(self, value: BaseEmbedder | None) -> None:
        """Set the default embedder (for backward compatibility)."""
        if value:
            self.embedder_manager.register_instance("default", value, set_as_default=True)

    # =========================================================================
    # Stage-based Model Configuration
    # =========================================================================

    def set_stage_model(self, stage: str, model: BaseLLM | BaseReranker | BaseRewriter) -> None:
        """
        Set the model for a specific stage.

        Args:
            stage: One of "chat", "rewrite", "router", "rerank", "qa_indexing"
            model: The model instance to use for this stage
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {self.STAGES}")

        self._stage_models[stage] = model

        # If setting chat model, also set as default LLM
        if stage == "chat" and isinstance(model, BaseLLM):
            self.default_llm = model

        logger.info(f"[RAGKernel] Stage '{stage}' model set: {model.__class__.__name__}")

    def get_stage_model(self, stage: str) -> BaseLLM | BaseReranker | BaseRewriter | None:
        """
        Get the model for a specific stage.
        Falls back to default_llm for LLM stages if not specifically configured.
        """
        model = self._stage_models.get(stage)
        if model is not None:
            return model

        # Fallback to default LLM for LLM-based stages
        if stage in ["chat", "rewrite", "router", "qa_indexing"]:
            return self.default_llm

        return None

    def get_stage_config(self) -> dict[str, str | None]:
        """Get configuration summary for all stages."""
        return {
            stage: model.__class__.__name__ if model else None
            for stage, model in self._stage_models.items()
        }

    @property
    def stage_config(self) -> dict[str, str | None]:
        """Get stage -> model_name mapping from model_manager (for backward compatibility)."""
        return self.model_manager.get_stage_config()

    # Convenience properties
    @property
    def reranker(self) -> BaseReranker | None:
        """Get the reranker model."""
        model = self.get_stage_model("rerank")
        return model if isinstance(model, BaseReranker) else None

    @property
    def rewriter(self) -> BaseRewriter | None:
        """Get or create the rewriter."""
        model = self.get_stage_model("rewrite")
        if isinstance(model, BaseRewriter):
            return model
        if isinstance(model, BaseLLM):
            return LLMRewriter(llm=model)
        return None

    @property
    def router(self) -> LLMRouter | None:
        """Get or create the router."""
        model = self.get_stage_model("router")
        if isinstance(model, LLMRouter):
            return model
        if isinstance(model, BaseLLM):
            return LLMRouter(llm=model)
        return None

    # =========================================================================
    # Simple Setters (for backward compatibility)
    # =========================================================================

    def set_embedder(
        self,
        embedder_or_type: BaseEmbedder | str,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        name: str = "default",
    ) -> None:
        """
        Register an embedder.

        Can be called with:
        - set_embedder(embedder_instance) - direct instance
        - set_embedder("openai", model="...", base_url="...", api_key="...") - factory creation

        Args:
            embedder_or_type: Either a BaseEmbedder instance or embedder type string
            model: Model name (for factory creation)
            base_url: Base URL (for factory creation)
            api_key: API key (for factory creation)
            name: Name to register the embedder under (default: "default")
        """
        if isinstance(embedder_or_type, BaseEmbedder):
            self.embedder_manager.register_instance(name, embedder_or_type, set_as_default=(name == "default"))
            logger.info(f"[RAGKernel] Embedder '{name}' set: {embedder_or_type.__class__.__name__}")
        else:
            # Register config for lazy instantiation
            config = {"model": model, "base_url": base_url, "api_key": api_key}
            self.embedder_manager.register(name, embedder_or_type, config, set_as_default=(name == "default"))
            logger.info(f"[RAGKernel] Embedder '{name}' registered via factory: {embedder_or_type}")

    def get_embedder(self, name: str | None = None) -> BaseEmbedder | None:
        """
        Get an embedder by name.

        Args:
            name: Embedder name. If None, returns the default embedder.

        Returns:
            BaseEmbedder instance or None if not found.
        """
        return self.embedder_manager.get(name)

    def set_llm(self, llm: BaseLLM) -> None:
        """Set the default LLM (also sets chat stage)."""
        self.default_llm = llm
        self.set_stage_model("chat", llm)

    def set_reranker(self, reranker: BaseReranker) -> None:
        """Set the reranker model."""
        self.set_stage_model("rerank", reranker)

    def add_llm(self, name: str, config: dict, set_as_default: bool = False) -> None:
        """
        Add a new LLM to the model pool.

        Args:
            name: Unique name for the model
            config: Model configuration dict
            set_as_default: Whether to set as default model
        """
        from web.core.factories import LLMFactory
        llm_instance = LLMFactory.create(config)
        self.model_manager.register_model(name, llm_instance, set_as_default)

        if set_as_default:
            self.default_llm = llm_instance

        logger.info(f"[RAGKernel] Added LLM '{name}' to pool")

    # =========================================================================
    # Vector Store Management
    # =========================================================================

    def create_vector_store(
        self,
        kb_id: str,
        collection_name: str,
        vdb_type: str = "duckdb",
        **kwargs
    ) -> BaseVector:
        """Create a new vector store for a knowledge base."""
        dataset = Dataset(
            id=kb_id,
            name=kwargs.get("name", collection_name),
            indexing_technique="high_quality",
            collection_name=collection_name,
            vdb_type=vdb_type,
        )
        store = self.vdb_manager.create_store(dataset)
        self._stores[kb_id] = store
        logger.info(f"[RAGKernel] Created vector store: {kb_id}")
        return store

    def get_vector_store(self, kb_id: str) -> BaseVector | None:
        """Get vector store by KB ID from cache."""
        return self._stores.get(kb_id)

    @property
    def vector_stores(self) -> dict[str, BaseVector]:
        """Alias for _stores (backward compatibility)."""
        return self._stores

    def delete_vector_store(self, kb_id: str) -> bool:
        """Delete a vector store."""
        store = self._stores.pop(kb_id, None)
        if store:
            try:
                store.delete()
                return True
            except Exception as e:
                logger.warning(f"[RAGKernel] Failed to delete store {kb_id}: {e}")
        return False

    # =========================================================================
    # RAG Operations (Delegates to langrag)
    # =========================================================================

    def process_document(
        self,
        file_path: Path,
        kb_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedder: BaseEmbedder | None = None,
        embedder_name: str | None = None,
    ) -> int:
        """
        Index a document into a knowledge base.

        Args:
            file_path: Path to document
            kb_id: Knowledge base ID
            chunk_size: Chunk size
            chunk_overlap: Overlap between chunks
            embedder: Runtime embedder override (takes precedence)
            embedder_name: Name of registered embedder to use

        Returns:
            Number of chunks stored
        """
        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found: {kb_id}")

        # Resolve embedder: runtime > named > default
        active_embedder = embedder or self.get_embedder(embedder_name)

        result = langrag.index_document_sync(
            file_path=file_path,
            vector_store=store,
            embedder=active_embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not result.success:
            raise RuntimeError(f"Indexing failed: {result.errors}")

        return result.stored_count

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
        rewriter: BaseRewriter | None = None,
        embedder: BaseEmbedder | None = None,
        embedder_name: str | None = None,
        use_rerank: bool | None = None,
        use_rewrite: bool | None = None,
        search_mode: str = "hybrid",
    ) -> tuple[list[LangRAGDocument], str, str | None]:
        """
        Search a knowledge base.

        Models are resolved in order:
        1. Runtime parameter (reranker=...)
        2. Stage-configured model
        3. None (disabled)

        Args:
            kb_id: Knowledge base ID
            query: Search query
            top_k: Number of results
            reranker: Runtime reranker override
            rewriter: Runtime rewriter override
            embedder: Runtime embedder override (takes precedence)
            embedder_name: Name of registered embedder to use
            use_rerank: Force enable/disable reranking
            use_rewrite: Force enable/disable rewriting
            search_mode: Search mode (hybrid, vector, keyword) - passed to vector store

        Returns:
            Tuple of (results, search_type, rewritten_query)
        """
        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found: {kb_id}")

        # Resolve embedder: runtime > named > default
        active_embedder = embedder or self.get_embedder(embedder_name)

        # Resolve models: runtime > stage config > None
        effective_reranker = reranker
        if effective_reranker is None and use_rerank is not False:
            effective_reranker = self.reranker if use_rerank else None

        effective_rewriter = rewriter
        if effective_rewriter is None and use_rewrite is not False:
            effective_rewriter = self.rewriter if use_rewrite else None

        result = langrag.search_sync(
            query=query,
            vector_stores=[store],
            embedder=active_embedder,
            reranker=effective_reranker,
            rewriter=effective_rewriter,
            top_k=top_k,
        )

        docs = [r.document for r in result.results]
        return docs, search_mode, result.rewritten_query

    def multi_search(
        self,
        kb_ids: list[str],
        query: str,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
        rewriter: BaseRewriter | None = None,
        embedder_name: str | None = None,
    ) -> tuple[list[LangRAGDocument], str]:
        """Search across multiple knowledge bases."""
        stores = [self.get_vector_store(kb_id) for kb_id in kb_ids]
        stores = [s for s in stores if s is not None]

        if not stores:
            return [], "none"

        # Resolve embedder
        active_embedder = self.get_embedder(embedder_name)

        result = langrag.search_sync(
            query=query,
            vector_stores=stores,
            embedder=active_embedder,
            reranker=reranker or self.reranker,
            rewriter=rewriter or self.rewriter,
            top_k=top_k,
        )

        docs = [r.document for r in result.results]
        return docs, "hybrid"

    async def chat(
        self,
        kb_ids: list[str],
        message: str,
        history: list[dict] | None = None,
        llm: BaseLLM | None = None,
        reranker: BaseReranker | None = None,
    ) -> str:
        """
        RAG-powered chat.

        Uses the "chat" stage model by default.

        Args:
            kb_ids: Knowledge bases to search
            message: User message
            history: Chat history
            llm: Runtime LLM override
            reranker: Runtime reranker override

        Returns:
            Generated response
        """
        stores = [self.get_vector_store(kb_id) for kb_id in kb_ids]
        stores = [s for s in stores if s is not None]

        active_llm = llm or self.get_stage_model("chat")
        if not active_llm or not isinstance(active_llm, BaseLLM):
            raise ValueError("No LLM configured for chat stage")

        result = await langrag.rag(
            query=message,
            vector_stores=stores,
            llm=active_llm,
            embedder=self.embedder,
            reranker=reranker or self.reranker,
            top_k=5,
        )

        return result.answer

    # =========================================================================
    # Cache Management
    # =========================================================================

    @property
    def cache_stats(self) -> dict | None:
        """Get cache statistics."""
        return self.cache.stats if self.cache else None

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self.cache:
            self.cache.clear()


# Global instance
rag_kernel = RAGKernel()
