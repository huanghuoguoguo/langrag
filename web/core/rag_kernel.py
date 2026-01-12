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
from langrag.retrieval.rewriter.base import BaseRewriter
from web.config import DATA_DIR, settings
from web.core.services.chat_service import ChatService
from web.core.services.document_processor import DocumentProcessor
from web.core.embedders import SeekDBEmbedder, WebOpenAIEmbedder
from web.core.services.retrieval_service import RetrievalService
from web.core.vdb_manager import WebVectorStoreManager
from web.core.kb_retrieval_config import KBRetrievalConfig, RerankerConfig, RewriterConfig

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
        - KB-level retrieval configuration support
        """
        # Core state
        self.embedder: BaseEmbedder | None = None
        self.reranker: BaseReranker | None = None  # 全局默认 reranker（向后兼容）
        self.vector_stores: dict[str, BaseVector] = {}
        self.kb_names: dict[str, str] = {}

        # ========== KB 级别配置 ==========
        # 每个知识库的检索配置
        self.kb_configs: dict[str, KBRetrievalConfig] = {}
        # Reranker 实例缓存（按配置缓存，避免重复创建）
        self._reranker_cache: dict[str, BaseReranker] = {}
        # Rewriter 实例缓存
        self._rewriter_cache: dict[str, BaseRewriter] = {}
        # Embedder 实例缓存（按名称缓存）
        self._embedder_cache: dict[str, BaseEmbedder] = {}

        # LLM configuration (Delegated to ModelManager)
        
        # Agentic components (Router, Rewriter) - 全局默认
        self.router = None
        self.rewriter = None
        
        # Model Management
        from web.core.model_manager import WebModelManager
        self.model_manager = WebModelManager()
        
        # Stage configuration (mapped to model names)
        self.stage_config = {
            "chat": None,     # Default chat model
            "router": None,   # Routing model
            "rewriter": None  # Query rewriting model
        }

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

    # =========================================================================
    # Model Management Methods
    # =========================================================================

    def add_llm(self, name: str, config: dict, set_as_default: bool = False) -> None:
        """
        Add a new LLM to the kernel's model pool.
        
        Delegates creation to LLMFactory.
        """
        try:
            from web.core.factories import LLMFactory
            
            # Create instance using factory
            llm_instance = LLMFactory.create(config)

            # Register with manager
            self.model_manager.register_model(name, llm_instance, set_as_default)
            
            # If this is the first model or default, update stages that are unset
            if set_as_default or list(self.model_manager.list_models()) == [name]:
                for stage in self.stage_config:
                    if self.stage_config[stage] is None:
                        self.set_stage_model(stage, name)
            
            logger.info(f"Added LLM '{name}' to pool.")

        except Exception as e:
            logger.error(f"Failed to add LLM '{name}': {e}")
            raise

    def set_stage_model(self, stage: str, model_name: str) -> None:
        """
        Assign a specific model to a pipeline stage.
        
        Args:
            stage: "chat", "router", or "rewriter"
            model_name: Name of a registered model
        """
        if stage not in self.stage_config:
            raise ValueError(f"Unknown stage: {stage}")
            
        model = self.model_manager.get_model(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found in manager.")
            
        self.stage_config[stage] = model_name
        
        # Re-initialize components based on stage changes
        if stage == "router":
            from langrag.retrieval.router.llm_router import LLMRouter
            self.router = LLMRouter(llm=model)
            logger.info(f"Router switched to use model: {model_name}")
            
        elif stage == "rewriter":
            from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
            self.rewriter = LLMRewriter(llm=model)
            logger.info(f"Rewriter switched to use model: {model_name}")
            
        elif stage == "chat":
            # Chat service is rebuilt lazily or via rebuild_services
            pass
            
        self._rebuild_services()

    def get_stage_model_name(self, stage: str) -> str | None:
        return self.stage_config.get(stage)

    # Legacy compatibility wrapper (DEPRECATED but kept for existing calls)
    def set_llm(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model_path: str | None = None 
    ) -> None:
        """
        Legacy method to set the default LLM.
        Directs to add_llm with name 'default'.
        """
        config = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if model_path:
            config["type"] = "local"
            config["model_path"] = model_path
            name = "default-local"
        else:
            config["type"] = "remote"
            config["base_url"] = base_url
            config["api_key"] = api_key
            config["model"] = model
            name = "default-remote"
            
        self.add_llm(name, config, set_as_default=True)

    def set_embedder(
        self,
        embedder_type: str,
        model: str = "",
        base_url: str = "",
        api_key: str = ""
    ) -> None:
        """
        Configure the embedding model.
        Delegates to EmbedderFactory.
        """
        config = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key
        }
        
        try:
            from web.core.factories import EmbedderFactory
            self.embedder = EmbedderFactory.create(embedder_type, config)
        except Exception as e:
            logger.error(f"Failed to set embedder: {e}")
            raise

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

    # =========================================================================
    # KB-Level Retrieval Configuration
    # =========================================================================

    def set_kb_retrieval_config(self, config: KBRetrievalConfig) -> None:
        """
        设置知识库的检索配置。
        
        Args:
            config: 知识库检索配置对象
        """
        self.kb_configs[config.kb_id] = config
        logger.info(f"[RAGKernel] KB retrieval config set for: {config.kb_id}")
        
        # 预创建 reranker 和 rewriter 实例（如果配置了）
        if config.reranker.enabled:
            self._get_or_create_reranker(config.reranker)
        if config.rewriter.enabled:
            self._get_or_create_rewriter(config.rewriter)

    def get_kb_retrieval_config(self, kb_id: str) -> KBRetrievalConfig | None:
        """
        获取知识库的检索配置。
        
        Args:
            kb_id: 知识库 ID
            
        Returns:
            检索配置对象，如果未配置则返回 None
        """
        return self.kb_configs.get(kb_id)

    def update_kb_retrieval_config(
        self,
        kb_id: str,
        search_mode: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        reranker_config: RerankerConfig | None = None,
        rewriter_config: RewriterConfig | None = None
    ) -> KBRetrievalConfig:
        """
        更新知识库的检索配置。
        
        Args:
            kb_id: 知识库 ID
            search_mode: 搜索模式
            top_k: 返回结果数量
            score_threshold: 分数阈值
            reranker_config: Reranker 配置
            rewriter_config: Rewriter 配置
            
        Returns:
            更新后的配置对象
        """
        config = self.kb_configs.get(kb_id)
        if not config:
            config = KBRetrievalConfig(kb_id=kb_id)
        
        if search_mode is not None:
            config.search_mode = search_mode
        if top_k is not None:
            config.top_k = top_k
        if score_threshold is not None:
            config.score_threshold = score_threshold
        if reranker_config is not None:
            config.reranker = reranker_config
        if rewriter_config is not None:
            config.rewriter = rewriter_config
            
        self.kb_configs[kb_id] = config
        logger.info(f"[RAGKernel] KB retrieval config updated for: {kb_id}")
        return config

    def _get_or_create_reranker(self, config: RerankerConfig) -> BaseReranker | None:
        """
        获取或创建 Reranker 实例（带缓存）。
        
        Args:
            config: Reranker 配置
            
        Returns:
            Reranker 实例，如果未启用则返回 None
        """
        if not config.enabled or not config.reranker_type:
            return None
            
        # 生成缓存键
        cache_key = f"{config.reranker_type}:{config.model or 'default'}:{config.api_key or 'none'}"
        
        if cache_key in self._reranker_cache:
            return self._reranker_cache[cache_key]
        
        try:
            params = {}
            if config.model:
                params["model"] = config.model
            if config.api_key:
                params["api_key"] = config.api_key
                
            reranker = RerankerFactory.create(config.reranker_type, **params)
            self._reranker_cache[cache_key] = reranker
            logger.info(f"[RAGKernel] Created reranker: {config.reranker_type}")
            return reranker
        except Exception as e:
            logger.error(f"[RAGKernel] Failed to create reranker: {e}")
            return None

    def _get_or_create_rewriter(self, config: RewriterConfig) -> BaseRewriter | None:
        """
        获取或创建 Rewriter 实例（带缓存）。
        
        Args:
            config: Rewriter 配置
            
        Returns:
            Rewriter 实例，如果未启用则返回 None
        """
        if not config.enabled or not config.llm_name:
            return None
            
        cache_key = config.llm_name
        
        if cache_key in self._rewriter_cache:
            return self._rewriter_cache[cache_key]
        
        try:
            llm = self.model_manager.get_model(config.llm_name)
            if not llm:
                logger.warning(f"[RAGKernel] LLM not found for rewriter: {config.llm_name}")
                return None
                
            from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
            rewriter = LLMRewriter(llm=llm)
            self._rewriter_cache[cache_key] = rewriter
            logger.info(f"[RAGKernel] Created rewriter with LLM: {config.llm_name}")
            return rewriter
        except Exception as e:
            logger.error(f"[RAGKernel] Failed to create rewriter: {e}")
            return None

    def get_reranker_for_kb(self, kb_id: str) -> BaseReranker | None:
        """
        获取指定知识库的 Reranker。
        
        优先使用 KB 级别配置，如果未配置则使用全局 Reranker。
        
        Args:
            kb_id: 知识库 ID
            
        Returns:
            Reranker 实例
        """
        config = self.kb_configs.get(kb_id)
        if config and config.reranker.enabled:
            return self._get_or_create_reranker(config.reranker)
        return self.reranker  # 回退到全局 reranker

    def get_rewriter_for_kb(self, kb_id: str) -> BaseRewriter | None:
        """
        获取指定知识库的 Rewriter。
        
        优先使用 KB 级别配置，如果未配置则使用全局 Rewriter。
        
        Args:
            kb_id: 知识库 ID
            
        Returns:
            Rewriter 实例
        """
        config = self.kb_configs.get(kb_id)
        if config and config.rewriter.enabled:
            return self._get_or_create_rewriter(config.rewriter)
        return self.rewriter  # 回退到全局 rewriter

    def get_embedder_for_kb(self, kb_id: str) -> BaseEmbedder | None:
        """
        获取指定知识库的 Embedder。
        
        优先使用 KB 级别配置的 embedder_name，如果未配置则使用全局 Embedder。
        
        Args:
            kb_id: 知识库 ID
            
        Returns:
            Embedder 实例
        """
        config = self.kb_configs.get(kb_id)
        if config and config.embedder_name:
            return self._get_or_create_embedder(config.embedder_name)
        return self.embedder  # 回退到全局 embedder

    def _get_or_create_embedder(self, embedder_name: str) -> BaseEmbedder | None:
        """
        获取或创建 Embedder 实例（带缓存）。
        
        从数据库加载 Embedder 配置并创建实例。
        
        Args:
            embedder_name: Embedder 配置名称
            
        Returns:
            Embedder 实例
        """
        if embedder_name in self._embedder_cache:
            return self._embedder_cache[embedder_name]
        
        try:
            # 从数据库加载配置
            from web.core.database import get_session
            from web.models.database import EmbedderConfig
            from sqlmodel import select
            
            session_gen = get_session()
            session = next(session_gen)
            try:
                statement = select(EmbedderConfig).where(EmbedderConfig.name == embedder_name)
                config = session.exec(statement).first()
                
                if not config:
                    logger.warning(f"[RAGKernel] Embedder config not found: {embedder_name}")
                    return self.embedder  # 回退到全局
                
                # 创建 Embedder 实例
                from web.core.factories import EmbedderFactory
                embedder = EmbedderFactory.create(
                    config.embedder_type,
                    {
                        "model": config.model,
                        "base_url": config.base_url,
                        "api_key": config.api_key
                    }
                )
                
                self._embedder_cache[embedder_name] = embedder
                logger.info(f"[RAGKernel] Created embedder: {embedder_name} ({config.embedder_type})")
                return embedder
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"[RAGKernel] Failed to create embedder '{embedder_name}': {e}")
            return self.embedder  # 回退到全局

    def _rebuild_services(self) -> None:
        """
        Rebuild services when dependencies change.
        """
        # Get default models
        chat_model = self.model_manager.get_model(self.stage_config.get("chat"))
        
        self._document_processor = DocumentProcessor(
            embedder=self.embedder,
            llm_adapter=chat_model,
            kv_store=self.kv_store
        )

        # Rebuild retrieval service (with semantic cache)
        self._retrieval_service = RetrievalService(
            embedder=self.embedder,
            reranker=self.reranker,
            rewriter=self.rewriter,
            kv_store=self.kv_store,
            cache=self.cache,
            vector_manager=self.vdb_manager
        )

        # Rebuild chat service using the CHAT stage model
        if chat_model:
            self._chat_service = ChatService(
                llm=chat_model,
                llm_config={}, 
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
        
        使用该知识库配置的 Embedder 进行向量化。
        """
        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")

        # 获取 KB 专属的 Embedder
        kb_embedder = self.get_embedder_for_kb(kb_id)
        
        logger.info(
            f"[RAGKernel] process_document: kb_id={kb_id}, "
            f"file={file_path}, technique={indexing_technique}, "
            f"embedder={kb_embedder.__class__.__name__ if kb_embedder else 'None'}"
        )

        # 为每个 KB 创建独立的 DocumentProcessor（使用该 KB 的 Embedder）
        chat_model = self.model_manager.get_model(self.stage_config.get("chat"))
        document_processor = DocumentProcessor(
            embedder=kb_embedder,
            llm_adapter=chat_model,
            kv_store=self.kv_store
        )

        return document_processor.process(
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
        top_k: int | None = None,
        search_mode: str | None = None,
        use_rerank: bool | None = None,
        use_rewrite: bool | None = None
    ) -> tuple[list[LangRAGDocument], str, str | None]:
        """
        Search a single knowledge base.

        优先使用 KB 级别的检索配置，参数可覆盖配置。

        Args:
            kb_id: The knowledge base to search
            query: The search query
            top_k: Number of results to return (None = use KB config or default 5)
            search_mode: Force search mode ("hybrid", "vector", "keyword") or None for KB config
            use_rerank: Force reranking on/off, or None for KB config
            use_rewrite: Whether to apply query rewriting, or None for KB config

        Returns:
            Tuple of (results list, search type string, rewritten query or None)

        Raises:
            ValueError: If the KB doesn't exist
        """
        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")

        # 获取 KB 级别配置
        kb_config = self.kb_configs.get(kb_id)
        
        # 合并配置：参数优先，否则使用 KB 配置，最后使用默认值
        effective_top_k = top_k if top_k is not None else (kb_config.top_k if kb_config else 5)
        effective_search_mode = search_mode if search_mode is not None else (kb_config.search_mode if kb_config else "hybrid")
        effective_score_threshold = kb_config.score_threshold if kb_config else 0.0
        
        # 确定是否使用 rerank
        if use_rerank is not None:
            effective_use_rerank = use_rerank
        elif kb_config and kb_config.reranker.enabled:
            effective_use_rerank = True
        else:
            effective_use_rerank = self.reranker is not None
            
        # 确定是否使用 rewrite
        if use_rewrite is not None:
            effective_use_rewrite = use_rewrite
        elif kb_config and kb_config.rewriter.enabled:
            effective_use_rewrite = True
        else:
            effective_use_rewrite = self.rewriter is not None

        logger.info(
            f"[RAGKernel] Search: kb_id={kb_id}, "
            f"query='{query[:50]}...', top_k={effective_top_k}, "
            f"mode={effective_search_mode}, rerank={effective_use_rerank}, rewrite={effective_use_rewrite}"
        )

        # 获取 KB 专属的组件
        kb_embedder = self.get_embedder_for_kb(kb_id)
        kb_reranker = self.get_reranker_for_kb(kb_id) if effective_use_rerank else None
        kb_rewriter = self.get_rewriter_for_kb(kb_id) if effective_use_rewrite else None

        # 创建临时的 RetrievalService（使用 KB 专属组件）
        retrieval_service = RetrievalService(
            embedder=kb_embedder,
            reranker=kb_reranker,
            rewriter=kb_rewriter,
            kv_store=self.kv_store,
            cache=self.cache,
            vector_manager=self.vdb_manager
        )

        # 获取 rerank_top_k
        rerank_top_k = None
        if kb_config and kb_config.reranker.top_k:
            rerank_top_k = kb_config.reranker.top_k

        return retrieval_service.search(
            store=store,
            query=query,
            top_k=effective_top_k,
            rewrite=effective_use_rewrite and kb_rewriter is not None,
            search_mode=effective_search_mode,
            use_rerank=effective_use_rerank,
            score_threshold=effective_score_threshold,
            rerank_top_k=rerank_top_k
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
        stream: bool = False,
        # Optional override: Use a specific model for THIS chat turn
        model_name: str | None = None
    ) -> dict | Any:
        # If model override provided, temporarily rebuild chat service or use ephemeral one
        service = self._chat_service
        
        if model_name:
            override_model = self.model_manager.get_model(model_name)
            if override_model:
                # Create ephemeral service for this request
                service = ChatService(
                    llm=override_model,
                    llm_config={},
                    retrieval_service=self._retrieval_service,
                    router=self.router,
                    kb_names=self.kb_names
                )
        
        if not service:
             # Try lazily init
             if not self._retrieval_service:
                self._retrieval_service = RetrievalService(
                    embedder=self.embedder,
                    reranker=self.reranker,
                    rewriter=self.rewriter,
                    kv_store=self.kv_store,
                    cache=self.cache,
                    vector_manager=self.vdb_manager
                )
                
             model = self.model_manager.get_model(self.stage_config.get("chat"))
             if model:
                self._chat_service = ChatService(
                    llm=model,
                    llm_config={},
                    retrieval_service=self._retrieval_service,
                    router=self.router,
                    kb_names=self.kb_names
                )
                service = self._chat_service

        if not service:
            raise ValueError("Chat service not initialized (no default chat model configured)")

        # Build stores dict from KB IDs
        kb_stores = {}
        for kb_id in kb_ids:
            store = self.get_vector_store(kb_id)
            if store:
                kb_stores[kb_id] = store

        return await service.chat(
            kb_stores=kb_stores,
            query=query,
            history=history,
            stream=stream
        )
