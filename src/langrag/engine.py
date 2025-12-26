"""RAG Engine - High-level orchestrator for RAG operations."""

from pathlib import Path
from loguru import logger

from .config.models import RAGConfig, StorageRole
from .config.factory import ComponentFactory
from .indexing import IndexingPipeline
from .retrieval import Retriever
from .core.search_result import SearchResult


class RAGEngine:
    """High-level orchestrator for RAG operations.

    This class manages the complete RAG system lifecycle, including:
    - Component initialization from configuration
    - Pipeline creation and management
    - High-level indexing and retrieval operations
    
    新特性：
    - 支持多数据源索引（角色分工）
    - 支持多数据源检索（能力自适应 + RRF 融合）
    - 向后兼容单数据源配置

    Attributes:
        config: RAG system configuration
        parser: Document parser
        chunker: Text chunker
        embedder: Embedding generator
        vector_stores: 向量存储列表（包含角色）
        reranker: Optional reranker
        llm: Optional LLM for generation
        indexing_pipeline: Pipeline for document indexing
        retrieval_pipeline: 检索管道（单源或多源）
    """

    def __init__(self, config: RAGConfig):
        """Initialize the RAG engine from configuration.

        Args:
            config: RAG system configuration

        Raises:
            ImportError: If component classes cannot be imported
            TypeError: If components don't match expected base classes
        """
        self.config = config
        logger.info("Initializing RAG Engine")

        # Create components
        self._create_components()

        # Create pipelines
        self._create_pipelines()

        logger.info("RAG Engine initialized successfully")

    def _create_components(self):
        """Create all components from configuration."""
        logger.debug("Creating components from configuration")

        self.parser = ComponentFactory.create_parser(self.config.parser)
        self.chunker = ComponentFactory.create_chunker(self.config.chunker)
        self.embedder = ComponentFactory.create_embedder(self.config.embedder)
        
        # 创建向量存储（支持单一或多个）
        vector_store_configs = self.config.get_vector_stores()
        
        if not vector_store_configs:
            raise ValueError("No vector store configured")
        
        self.vector_stores = []
        for vs_config in vector_store_configs:
            store = ComponentFactory.create_vector_store(vs_config)
            role = vs_config.role if hasattr(vs_config, 'role') else StorageRole.PRIMARY
            self.vector_stores.append((store, role))
            logger.info(f"Created vector store: {store.__class__.__name__} (role={role.value})")

        # Optional components
        self.reranker = None
        if self.config.reranker:
            self.reranker = ComponentFactory.create_reranker(
                self.config.reranker
            )

        self.llm = None
        if self.config.llm:
            self.llm = ComponentFactory.create_llm(self.config.llm)

        logger.debug("All components created")

    def _create_pipelines(self):
        """Create indexing and retrieval pipelines."""
        logger.debug("Creating pipelines")

        # 1. 索引管道：支持多存储写入
        self.indexing_pipeline = IndexingPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_stores=self.vector_stores,  # 传入列表
        )

        # 2. 检索管道：根据配置选择单源或多源模式
        retrieval_config = self.config.get_retrieval_config()
        
        if retrieval_config.mode == "single" or (
            retrieval_config.mode == "auto" and len(self.vector_stores) == 1
        ):
            # 单源模式（能力自适应）
            vector_store, storage_role = self.vector_stores[0]
            logger.info(f"Using single-store retrieval mode: {vector_store.__class__.__name__}")
            
            # 创建 Retriever（能力自适应）
            retriever = Retriever.from_single_store(
                embedder=self.embedder,
                vector_store=vector_store,
                storage_role=storage_role
            )
            
            # 使用 Retriever 创建检索管道
            self.retrieval_pipeline = AdaptiveRetrievalPipeline(
                retriever=retriever,
                reranker=self.reranker,
                top_k=retrieval_config.final_top_k,
                rerank_top_k=self.config.rerank_top_k,
            )
            
        elif retrieval_config.mode == "multi_store" or (
            retrieval_config.mode == "auto" and len(self.vector_stores) > 1
        ):
            # 多源模式（角色分工 + RRF 融合）
            logger.info(f"Using multi-store retrieval mode with {len(self.vector_stores)} stores")
            
            # 创建 Retriever（多源配置）
            retriever = Retriever.from_multi_stores(
            embedder=self.embedder,
                stores_config=self.vector_stores,
                fusion_strategy=retrieval_config.fusion_strategy,
                fusion_weights=retrieval_config.fusion_weights,
            )
            
            # 使用 Retriever 创建检索管道
            self.retrieval_pipeline = AdaptiveRetrievalPipeline(
                retriever=retriever,
            reranker=self.reranker,
                top_k=retrieval_config.final_top_k,
            rerank_top_k=self.config.rerank_top_k,
        )
            
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_config.mode}")

        logger.debug("Pipelines created")

    def index(self, file_path: str | Path) -> int:
        """Index a single file.

        Args:
            file_path: Path to file to index

        Returns:
            Number of chunks indexed

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        return self.indexing_pipeline.index_file(file_path)

    def index_batch(self, file_paths: list[str | Path]) -> int:
        """Index multiple files.

        Args:
            file_paths: List of file paths to index

        Returns:
            Total number of chunks indexed
        """
        return self.indexing_pipeline.index_files(file_paths)

    def retrieve(self, query: str) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string

        Returns:
            List of search results, sorted by relevance
        """
        return self.retrieval_pipeline.retrieve(query)

    def query(self, query: str, use_llm: bool = True) -> str | list[SearchResult]:
        """Query the RAG system.

        If LLM is configured and use_llm=True, generates a response.
        Otherwise, returns raw retrieval results.

        Args:
            query: Query string
            use_llm: Whether to use LLM for generation (default: True)

        Returns:
            Generated response (if LLM available) or list of search results
        """
        results = self.retrieve(query)

        if use_llm and self.llm is not None:
            logger.info("Generating response with LLM")
            return self.llm.generate(query=query, context=results)

        return results

    def save_index(self, path: str):
        """Save the vector store index to disk.

        Args:
            path: File path to save to
        """
        logger.info(f"Saving index to {path}")
        self.vector_store.persist(path)

    def load_index(self, path: str):
        """Load the vector store index from disk.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        logger.info(f"Loading index from {path}")
        self.vector_store.load(path)

    @property
    def num_chunks(self) -> int:
        """Get the number of chunks in the vector store.

        Note: This requires the vector store to expose chunk count.
        May not be available for all implementations.
        For multi-store setup, returns count from the first PRIMARY store.
        """
        # 找到第一个 PRIMARY 存储
        for store, role in self.vector_stores:
            if role == StorageRole.PRIMARY:
                if hasattr(store, "_chunks"):
                    return len(store._chunks)
                break
        
        # 如果没有 PRIMARY，使用第一个存储
        if self.vector_stores and hasattr(self.vector_stores[0][0], "_chunks"):
            return len(self.vector_stores[0][0]._chunks)
        
        return 0


class AdaptiveRetrievalPipeline:
    """自适应检索管道
    
    使用 Retriever 进行检索，支持：
    - 单源检索（能力自适应）
    - 多源检索（RRF 融合）
    
    与原 RetrievalPipeline 兼容的接口。
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker=None,
        top_k: int = 5,
        rerank_top_k: int | None = None,
    ):
        """初始化自适应检索管道
        
        Args:
            retriever: 检索协调器
            reranker: 可选的重排序器
            top_k: 检索结果数
            rerank_top_k: 重排序后返回的结果数
        """
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query_text: str) -> list[SearchResult]:
        """同步检索接口（向后兼容）
        
        Args:
            query_text: 查询文本
            
        Returns:
            检索结果列表
        """
        import asyncio
        
        # 在同步上下文中运行异步检索
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.retrieve_async(query_text))

    async def retrieve_async(self, query_text: str) -> list[SearchResult]:
        """异步检索（推荐使用）
        
        Args:
            query_text: 查询文本
            
        Returns:
            检索结果列表
        """
        logger.info(f"Retrieving for query: {query_text[:50]}...")
        
        # 1. 使用 Retriever 检索
        results = await self.retriever.retrieve(query_text, self.top_k)
        logger.debug(f"Retrieved {len(results)} results")
        
        # 2. 可选的重排序
        if self.reranker is not None:
            from .core.query import Query
            query = Query(text=query_text, vector=None)  # Reranker 通常不需要 vector
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=self.rerank_top_k
            )
            logger.debug(f"Reranking returned {len(results)} results")
        
        return results
