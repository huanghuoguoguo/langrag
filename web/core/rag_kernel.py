"""
RAG Kernel - The central coordinator for the LangRAG Web Application.

This module serves as the main entry point and coordinator for all RAG operations.
It demonstrates how to integrate LangRAG core components into a web application
by managing configuration, dependencies, and service orchestration.

Architecture Overview:
----------------------
The RAG Kernel follows a service-oriented architecture where each concern
is handled by a dedicated service:

    RAGKernel (Coordinator)
        ├── WebVectorStoreManager (VDB lifecycle)
        ├── DocumentProcessor (Indexing pipeline)
        ├── RetrievalService (Search operations)
        └── ChatService (Conversational AI)

This separation provides:
- Single Responsibility: Each service handles one concern
- Testability: Services can be unit tested independently
- Flexibility: Services can be swapped or extended
- Clarity: Clear boundaries between components

Design Decisions:
-----------------
1. **Lazy Service Initialization**: Services are created on-demand when their
   dependencies become available (e.g., ChatService after LLM is configured).

2. **Configuration Delegation**: The kernel provides high-level configuration
   methods (set_llm, set_embedder) that propagate to relevant services.

3. **Backward Compatibility**: The public API remains unchanged from the
   original monolithic implementation to avoid breaking existing code.

4. **Dependency Injection**: Core LangRAG components (Router, Rewriter, Reranker)
   are created and injected into services, enabling Agentic RAG features.

Example Usage:
--------------
    # Initialize kernel
    kernel = RAGKernel()

    # Configure components
    kernel.set_embedder("openai", model="text-embedding-3-small", ...)
    kernel.set_llm(base_url="...", api_key="...", model="gpt-4")
    kernel.set_reranker("cohere", api_key="...")

    # Create knowledge base
    store = kernel.create_vector_store(
        kb_id="my-kb",
        collection_name="my_collection",
        vdb_type="chroma"
    )

    # Process documents
    kernel.process_document(
        file_path=Path("document.pdf"),
        kb_id="my-kb",
        indexing_technique="high_quality"
    )

    # Search
    results, search_type = kernel.search("my-kb", "What is X?")

    # Chat
    response = await kernel.chat(["my-kb"], "Explain X in detail")
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add src to python path for LangRAG imports
sys.path.append(str(Path(__file__).parents[2] / "src"))

from langrag import BaseEmbedder, BaseVector, Dataset, EmbedderFactory
from langrag import Document as LangRAGDocument
from langrag.cache import SemanticCache
from langrag.datasource.kv.sqlite import SQLiteKV
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rerank.factory import RerankerFactory
from web.config import DATA_DIR, settings
from web.core.chat_service import ChatService
from web.core.document_processor import DocumentProcessor
from web.core.embedders import SeekDBEmbedder, WebOpenAIEmbedder
from web.core.retrieval_service import RetrievalService
from web.core.vdb_manager import WebVectorStoreManager

logger = logging.getLogger(__name__)


class RAGKernel:
    """
    Central coordinator for RAG operations in the web application.

    The RAGKernel is the main integration point between your web application
    and the LangRAG library. It manages:

    - Component Configuration: Embedders, LLMs, Rerankers
    - Vector Store Lifecycle: Creation, caching, deletion
    - Document Processing: Parsing, chunking, indexing
    - Retrieval: Single and multi-KB search
    - Chat: RAG-powered conversational AI

    The kernel uses a delegation pattern where specialized services handle
    specific operations. This makes the codebase maintainable and testable
    while providing a simple, unified API.

    Attributes:
        embedder: Currently configured embedding model
        reranker: Currently configured reranking model
        vector_stores: Cache of active vector stores by KB ID
        kb_names: Human-readable names for knowledge bases
    """

    def __init__(self):
        """
        Initialize the RAG Kernel with default configuration.

        Sets up:
        - Vector store manager for VDB lifecycle management
        - SQLite KV store for parent-child indexing
        - Semantic cache for query deduplication (if enabled)
        - Empty service references (configured lazily)
        - Custom embedder registration
        """
        # Core state
        self.embedder: BaseEmbedder | None = None
        self.reranker: BaseReranker | None = None
        self.vector_stores: dict[str, BaseVector] = {}
        self.kb_names: dict[str, str] = {}

        # LLM configuration
        self.llm_client = None
        self.llm_config: dict[str, Any] = {}
        self.llm_adapter = None

        # Agentic components (Router, Rewriter)
        self.router = None
        self.rewriter = None

        # Initialize managers and stores
        self.vdb_manager = WebVectorStoreManager()

        # SQLite KV for persistent parent-child storage
        kv_path = DATA_DIR / "kv_store.sqlite"
        self.kv_store = SQLiteKV(db_path=str(kv_path))
        logger.info(f"[RAGKernel] KV Store initialized at {kv_path}")

        # Semantic cache (configuration-driven)
        self.cache: SemanticCache | None = None
        if settings.CACHE_ENABLED:
            self.cache = SemanticCache(
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
                ttl_seconds=settings.CACHE_TTL_SECONDS,
                max_size=settings.CACHE_MAX_SIZE
            )
            logger.info(
                f"[RAGKernel] Semantic cache enabled: "
                f"threshold={settings.CACHE_SIMILARITY_THRESHOLD}, "
                f"ttl={settings.CACHE_TTL_SECONDS}s, "
                f"max_size={settings.CACHE_MAX_SIZE}"
            )

        # Services (initialized lazily when dependencies are ready)
        self._document_processor: DocumentProcessor | None = None
        self._retrieval_service: RetrievalService | None = None
        self._chat_service: ChatService | None = None

        # Register custom embedder type for factory-based creation
        try:
            EmbedderFactory.register("web_openai", WebOpenAIEmbedder)
        except Exception:
            pass  # Already registered

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def set_llm(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> None:
        """
        Configure the LLM for chat and agentic components.

        This method sets up:
        1. AsyncOpenAI client for chat API calls
        2. WebLLMAdapter for LangRAG core compatibility
        3. LLMRouter for intelligent KB selection
        4. LLMRewriter for query optimization

        Args:
            base_url: API endpoint (e.g., "https://api.openai.com/v1")
            api_key: API authentication key
            model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0-2.0, default: 0.7)
            max_tokens: Maximum response tokens (default: 2048)

        Note:
            After calling this method, the ChatService will be available
            and Agentic RAG features (routing, rewriting) will be enabled.
        """
        try:
            from openai import AsyncOpenAI

            from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
            from langrag.retrieval.router.llm_router import LLMRouter
            from web.core.llm_adapter import WebLLMAdapter

            # Configure LLM client
            self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.llm_config = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            logger.info(f"LLM configured: {model} (base_url={base_url})")

            # Create adapter for LangRAG core components
            self.llm_adapter = WebLLMAdapter(self.llm_client, model=model)

            # Initialize Agentic RAG components
            self.router = LLMRouter(llm=self.llm_adapter)
            self.rewriter = LLMRewriter(llm=self.llm_adapter)
            logger.info("Agentic components initialized (Router, Rewriter)")

            # Rebuild services with new configuration
            self._rebuild_services()

        except ImportError:
            logger.error("openai package not installed. Cannot configure LLM.")
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")
            import traceback
            traceback.print_exc()

    def set_embedder(
        self,
        embedder_type: str,
        model: str = "",
        base_url: str = "",
        api_key: str = ""
    ) -> None:
        """
        Configure the embedding model.

        Supported embedder types:
        - "openai": OpenAI-compatible API (requires base_url, api_key, model)
        - "seekdb": Local all-MiniLM-L6-v2 via pyseekdb (no config needed)

        Args:
            embedder_type: Type of embedder ("openai" or "seekdb")
            model: Model name for API-based embedders
            base_url: API endpoint for API-based embedders
            api_key: API key for API-based embedders

        Raises:
            ValueError: If embedder_type is unsupported or required args missing
        """
        if embedder_type == "openai":
            if not base_url or not api_key or not model:
                raise ValueError("OpenAI embedder requires base_url, api_key and model")
            self.embedder = WebOpenAIEmbedder(base_url, api_key, model)
            logger.info(f"OpenAI-compatible embedder configured: {model}")

        elif embedder_type == "seekdb":
            self.embedder = SeekDBEmbedder()
            logger.info("SeekDB embedder configured (all-MiniLM-L6-v2)")

        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")

        # Rebuild services with new embedder
        self._rebuild_services()

    def set_reranker(self, provider: str, **kwargs) -> None:
        """
        Configure the reranking model.

        Reranking improves retrieval quality by using a cross-encoder
        to score query-document pairs more accurately than embedding similarity.

        Args:
            provider: Reranker provider (e.g., "cohere", "noop")
            **kwargs: Provider-specific configuration (e.g., api_key, model)

        Raises:
            Exception: If reranker creation fails
        """
        logger.info(f"[RAGKernel] Setting reranker: {provider}")
        try:
            self.reranker = RerankerFactory.create(provider, **kwargs)
            self._rebuild_services()
        except Exception as e:
            logger.error(f"Failed to set reranker: {e}")
            raise

    def set_cache(
        self,
        enabled: bool = True,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 1000
    ) -> None:
        """
        Configure the semantic cache programmatically.

        This method allows overriding the configuration-driven cache settings
        at runtime. Useful for testing or dynamic cache configuration.

        Args:
            enabled: Whether to enable caching (True) or disable it (False)
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            ttl_seconds: Cache entry lifetime in seconds (0 = no expiration)
            max_size: Maximum cached entries (0 = unlimited)

        Example:
            >>> kernel.set_cache(enabled=True, similarity_threshold=0.98)
            >>> kernel.set_cache(enabled=False)  # Disable caching
        """
        if enabled:
            self.cache = SemanticCache(
                similarity_threshold=similarity_threshold,
                ttl_seconds=ttl_seconds,
                max_size=max_size
            )
            logger.info(
                f"[RAGKernel] Semantic cache configured: "
                f"threshold={similarity_threshold}, ttl={ttl_seconds}s, max_size={max_size}"
            )
        else:
            self.cache = None
            logger.info("[RAGKernel] Semantic cache disabled")

        self._rebuild_services()

    @property
    def cache_stats(self) -> dict | None:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, etc.
            None if cache is disabled.

        Example:
            >>> stats = kernel.cache_stats
            >>> if stats:
            ...     print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        if self.cache:
            return self.cache.stats
        return None

    def clear_cache(self) -> None:
        """Clear all cached query results."""
        if self.cache:
            self.cache.clear()
            logger.info("[RAGKernel] Cache cleared")

    def _rebuild_services(self) -> None:
        """
        Rebuild services when dependencies change.

        This ensures services always have the latest configuration.
        Called automatically by set_llm, set_embedder, and set_reranker.
        """
        # Rebuild document processor
        self._document_processor = DocumentProcessor(
            embedder=self.embedder,
            llm_adapter=self.llm_adapter,
            kv_store=self.kv_store
        )

        # Rebuild retrieval service (with semantic cache)
        self._retrieval_service = RetrievalService(
            embedder=self.embedder,
            reranker=self.reranker,
            rewriter=self.rewriter,
            kv_store=self.kv_store,
            cache=self.cache
        )

        # Rebuild chat service (only if LLM is configured)
        if self.llm_client:
            self._chat_service = ChatService(
                llm_client=self.llm_client,
                llm_config=self.llm_config,
                retrieval_service=self._retrieval_service,
                router=self.router,
                kb_names=self.kb_names
            )

    # =========================================================================
    # Vector Store Management
    # =========================================================================

    def create_vector_store(
        self,
        kb_id: str,
        collection_name: str,
        vdb_type: str,
        name: str | None = None
    ) -> BaseVector:
        """
        Create a vector store for a knowledge base.

        This creates a new vector store instance and registers it
        for use in retrieval and chat operations.

        Args:
            kb_id: Unique identifier for the knowledge base
            collection_name: Name for the vector collection
            vdb_type: Vector database type ("chroma", "duckdb", "seekdb")
            name: Human-readable name (defaults to kb_id)

        Returns:
            The created vector store instance
        """
        dataset = Dataset(
            id=kb_id,
            tenant_id="default",
            name=name or kb_id,
            description="",
            indexing_technique="high_quality",
            collection_name=collection_name,
            vdb_type=vdb_type
        )

        store = self.vdb_manager.create_store(dataset)

        # Register in local tracking
        self.vector_stores[kb_id] = store
        self.kb_names[kb_id] = name or kb_id

        logger.info(f"[RAGKernel] Vector store created: kb_id={kb_id}, type={vdb_type}")
        return store

    def get_vector_store(self, kb_id: str) -> BaseVector | None:
        """
        Get a vector store by KB ID.

        Args:
            kb_id: The knowledge base identifier

        Returns:
            The vector store instance, or None if not found
        """
        if kb_id in self.vector_stores:
            return self.vector_stores[kb_id]

        # Try manager as fallback
        if hasattr(self.vdb_manager, 'get_store_by_id'):
            return self.vdb_manager.get_store_by_id(kb_id)

        return None

    # =========================================================================
    # Document Processing
    # =========================================================================

    def process_document(
        self,
        file_path: Path,
        kb_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        indexing_technique: str = "high_quality"
    ) -> int:
        """
        Process a document into the knowledge base.

        This method handles the complete indexing pipeline:
        1. Parse the document based on file type
        2. Chunk the content using the specified technique
        3. Generate embeddings (if embedder configured)
        4. Store in the vector database

        Args:
            file_path: Path to the document file
            kb_id: Target knowledge base ID
            chunk_size: Maximum characters per chunk (default: 500)
            chunk_overlap: Character overlap between chunks (default: 50)
            indexing_technique: Indexing strategy:
                - "high_quality": Standard chunking (default)
                - "qa": QA-pair generation
                - "parent_child": Hierarchical chunks

        Returns:
            Number of chunks/items created

        Raises:
            ValueError: If the KB doesn't exist or required deps missing
        """
        logger.info(
            f"[RAGKernel] process_document: kb_id={kb_id}, "
            f"file={file_path}, technique={indexing_technique}"
        )

        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")

        # Ensure document processor exists
        if not self._document_processor:
            self._document_processor = DocumentProcessor(
                embedder=self.embedder,
                llm_adapter=self.llm_adapter,
                kv_store=self.kv_store
            )

        return self._document_processor.process(
            file_path=file_path,
            vector_store=store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            indexing_technique=indexing_technique
        )

    # =========================================================================
    # Retrieval
    # =========================================================================

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        search_mode: str | None = None,
        use_rerank: bool | None = None,
        use_rewrite: bool = True
    ) -> tuple[list[LangRAGDocument], str]:
        """
        Search a single knowledge base.

        Args:
            kb_id: The knowledge base to search
            query: The search query
            top_k: Number of results to return (default: 5)
            search_mode: Force search mode ("hybrid", "vector", "keyword") or None for auto
            use_rerank: Force reranking on/off, or None for default (use if configured)
            use_rewrite: Whether to apply query rewriting (default: True)

        Returns:
            Tuple of (results list, search type string)

        Raises:
            ValueError: If the KB doesn't exist
        """
        logger.info(
            f"[RAGKernel] Search: kb_id={kb_id}, "
            f"query='{query[:50]}...', top_k={top_k}, mode={search_mode}"
        )

        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")

        # Ensure retrieval service exists
        if not self._retrieval_service:
            self._retrieval_service = RetrievalService(
                embedder=self.embedder,
                reranker=self.reranker,
                rewriter=self.rewriter,
                kv_store=self.kv_store,
                cache=self.cache
            )

        return self._retrieval_service.search(
            store=store,
            query=query,
            top_k=top_k,
            rewrite=use_rewrite and self.rewriter is not None,
            search_mode=search_mode,
            use_rerank=use_rerank
        )

    def multi_search(
        self,
        kb_ids: list[str],
        query: str,
        top_k: int = 5
    ) -> tuple[list[LangRAGDocument], str]:
        """
        Search across multiple knowledge bases.

        Args:
            kb_ids: List of knowledge base IDs to search
            query: The search query
            top_k: Number of results to return (default: 5)

        Returns:
            Tuple of (results list, search type string)
        """
        # Build stores dict from KB IDs
        stores = {}
        for kb_id in kb_ids:
            store = self.get_vector_store(kb_id)
            if store:
                stores[kb_id] = store
            else:
                logger.warning(f"Vector store not found for kb_id: {kb_id}")

        if not stores:
            return [], "none"

        # Ensure retrieval service exists
        if not self._retrieval_service:
            self._retrieval_service = RetrievalService(
                embedder=self.embedder,
                reranker=self.reranker,
                rewriter=self.rewriter,
                kv_store=self.kv_store,
                cache=self.cache
            )

        return self._retrieval_service.multi_search(
            stores=stores,
            query=query,
            top_k=top_k,
            rewrite=self.rewriter is not None
        )

    # =========================================================================
    # Chat
    # =========================================================================

    async def chat(
        self,
        kb_ids: list[str],
        query: str,
        history: list[dict] | None = None,
        stream: bool = False
    ) -> dict | Any:
        """
        Execute RAG-powered chat.

        Args:
            kb_ids: Knowledge bases to use for context (empty for no RAG)
            query: The user's question
            history: Previous conversation messages
            stream: Whether to stream the response

        Returns:
            If stream=False: {"answer": "...", "sources": [...]}
            If stream=True: AsyncGenerator yielding JSONL chunks

        Raises:
            ValueError: If LLM is not configured
        """
        if not self.llm_client:
            raise ValueError("LLM is not configured")

        # Build stores dict from KB IDs
        kb_stores = {}
        for kb_id in kb_ids:
            store = self.get_vector_store(kb_id)
            if store:
                kb_stores[kb_id] = store

        # Ensure chat service exists
        if not self._chat_service:
            if not self._retrieval_service:
                self._retrieval_service = RetrievalService(
                    embedder=self.embedder,
                    reranker=self.reranker,
                    rewriter=self.rewriter,
                    kv_store=self.kv_store,
                    cache=self.cache
                )

            self._chat_service = ChatService(
                llm_client=self.llm_client,
                llm_config=self.llm_config,
                retrieval_service=self._retrieval_service,
                router=self.router,
                kb_names=self.kb_names
            )

        return await self._chat_service.chat(
            kb_stores=kb_stores,
            query=query,
            history=history,
            stream=stream
        )
