"""RAG Engine - High-level orchestrator for RAG operations."""

from pathlib import Path
from loguru import logger

from .config.models import RAGConfig, StorageRole
from .config.factory import ComponentFactory
from .indexing import IndexingPipeline
from .retrieval import Retriever
from .core.search_result import SearchResult
from .utils import run_async_in_sync_context


class RAGEngine:
    """High-level orchestrator for RAG operations.

    This class manages the complete RAG system lifecycle, including:
    - Component initialization from configuration
    - Pipeline creation and management
    - High-level indexing and retrieval operations

    Features:
    - Multi-store indexing with role-based storage
    - Adaptive retrieval (single/multi-store with automatic strategy selection)
    - Built-in reranking support
    - Backward compatible with single-store configurations

    Attributes:
        config: RAG system configuration
        parser: Document parser
        chunker: Text chunker
        embedder: Embedding generator
        vector_stores: List of (vector_store, role) tuples
        reranker: Optional reranker
        llm: Optional LLM for generation
        indexing_pipeline: Pipeline for document indexing
        retriever: Retrieval coordinator (replaces retrieval_pipeline)
    """

    def __init__(self, config: RAGConfig):
        """Initialize the RAG engine from configuration.

        Args:
            config: RAG system configuration

        Raises:
            ValueError: If configuration is invalid
            ImportError: If component classes cannot be imported
            TypeError: If components don't match expected base classes
        """
        self.config = config
        logger.info("Initializing RAG Engine")

        # Validate configuration
        self._validate_config()

        # Create components
        self._create_components()

        # Create pipelines
        self._create_pipelines()

        logger.info("RAG Engine initialized successfully")

    def _validate_config(self) -> None:
        """Validate RAG configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required components
        if not self.config.parser:
            raise ValueError("Parser configuration is required")

        if not self.config.chunker:
            raise ValueError("Chunker configuration is required")

        if not self.config.embedder:
            raise ValueError("Embedder configuration is required")

        # Check vector stores
        vector_stores = self.config.get_vector_stores()
        if not vector_stores:
            raise ValueError("At least one vector store must be configured")

        # Check retrieval configuration
        retrieval_config = self.config.get_retrieval_config()
        if retrieval_config.mode not in {"single", "multi_store", "auto"}:
            raise ValueError(f"Invalid retrieval mode: {retrieval_config.mode}")

        # Check fusion weights
        if retrieval_config.fusion_weights:
            if len(retrieval_config.fusion_weights) != len(vector_stores):
                raise ValueError(
                    f"Fusion weights count ({len(retrieval_config.fusion_weights)}) "
                    f"must match vector stores count ({len(vector_stores)})"
                )

        logger.debug("Configuration validation passed")

    def _create_components(self) -> None:
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

    def _create_pipelines(self) -> None:
        """Create indexing and retrieval pipelines."""
        logger.debug("Creating pipelines")

        # 1. Create indexing pipeline
        self.indexing_pipeline = self._create_indexing_pipeline()

        # 2. Create retriever
        retrieval_config = self.config.get_retrieval_config()
        self.retriever = self._create_retriever(retrieval_config)

        # 3. Store configuration for later use
        self._retrieval_top_k = retrieval_config.final_top_k
        self._rerank_top_k = self.config.rerank_top_k

        logger.debug("Pipelines created")

    def _create_indexing_pipeline(self) -> IndexingPipeline:
        """Create indexing pipeline.

        Returns:
            Configured IndexingPipeline instance
        """
        return IndexingPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_stores=self.vector_stores,
        )

    def _create_retriever(self, retrieval_config) -> Retriever:
        """Create retriever based on configuration.

        Args:
            retrieval_config: Retrieval configuration

        Returns:
            Configured Retriever instance

        Raises:
            ValueError: If retrieval mode is unknown
        """
        mode = retrieval_config.mode
        num_stores = len(self.vector_stores)

        # Single-store mode
        if mode == "single" or (mode == "auto" and num_stores == 1):
            return self._create_single_store_retriever()

        # Multi-store mode
        if mode == "multi_store" or (mode == "auto" and num_stores > 1):
            return self._create_multi_store_retriever(retrieval_config)

        # Unknown mode
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def _create_single_store_retriever(self) -> Retriever:
        """Create single-store retriever with capability adaptation.

        Returns:
            Configured Retriever instance
        """
        vector_store, storage_role = self.vector_stores[0]
        logger.info(
            f"Using single-store retrieval mode: {vector_store.__class__.__name__}"
        )

        return Retriever.from_single_store(
            embedder=self.embedder,
            vector_store=vector_store,
            storage_role=storage_role,
            reranker=self.reranker
        )

    def _create_multi_store_retriever(self, retrieval_config) -> Retriever:
        """Create multi-store retriever with RRF fusion.

        Args:
            retrieval_config: Retrieval configuration

        Returns:
            Configured Retriever instance
        """
        logger.info(
            f"Using multi-store retrieval mode with {len(self.vector_stores)} stores"
        )

        return Retriever.from_multi_stores(
            embedder=self.embedder,
            stores_config=self.vector_stores,
            reranker=self.reranker,
            fusion_strategy=retrieval_config.fusion_strategy,
            fusion_weights=retrieval_config.fusion_weights,
        )

    def index(self, file_path: str | Path) -> int:
        """Index a single file.

        Args:
            file_path: Path to file to index

        Returns:
            Number of chunks indexed

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or file_path is empty
        """
        if not file_path:
            raise ValueError("file_path cannot be empty")

        return self.indexing_pipeline.index_file(file_path)

    def index_batch(self, file_paths: list[str | Path]) -> int:
        """Index multiple files.

        Args:
            file_paths: List of file paths to index

        Returns:
            Total number of chunks indexed

        Raises:
            ValueError: If file_paths is empty or not a list
        """
        if not file_paths:
            raise ValueError("file_paths cannot be empty")

        if not isinstance(file_paths, list):
            raise ValueError("file_paths must be a list")

        return self.indexing_pipeline.index_files(file_paths)

    def retrieve(self, query: str) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        This is a synchronous wrapper around retrieve_async().
        For async environments (Jupyter, FastAPI), use retrieve_async() directly.

        Args:
            query: Query string

        Returns:
            List of search results, sorted by relevance

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        return run_async_in_sync_context(self.retrieve_async(query))

    async def retrieve_async(self, query: str) -> list[SearchResult]:
        """Retrieve relevant chunks for a query (async version).

        This is the recommended method for async environments.

        Args:
            query: Query string

        Returns:
            List of search results, sorted by relevance

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        return await self.retriever.retrieve(
            query=query,
            top_k=self._retrieval_top_k,
            rerank_top_k=self._rerank_top_k
        )

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

    def save_index(self, path: str | Path):
        """Save all vector store indexes to disk.

        For single-store setup, saves directly to the specified path.
        For multi-store setup, creates a directory and saves each store
        to a subdirectory named by store class and role.

        Args:
            path: Directory path to save to (will be created if needed)
        """
        path = Path(path)

        if len(self.vector_stores) == 1:
            # Single store: save directly to path
            store, role = self.vector_stores[0]
            logger.info(f"Saving single store to {path}")
            store.persist(str(path))
        else:
            # Multi-store: create directory and save each store
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {len(self.vector_stores)} stores to {path}")

            for i, (store, role) in enumerate(self.vector_stores):
                store_name = f"{store.__class__.__name__}_{role.value}_{i}"
                store_path = path / store_name

                try:
                    store.persist(str(store_path))
                    logger.info(f"  ✓ Saved {store_name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to save {store_name}: {e}")
                    # Continue saving other stores

        logger.info(f"Index save completed: {path}")

    def load_index(self, path: str | Path):
        """Load vector store indexes from disk.

        For single-store setup, loads directly from the specified path.
        For multi-store setup, loads each store from its subdirectory.

        Args:
            path: Directory path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")

        if len(self.vector_stores) == 1:
            # Single store: load directly from path
            store, role = self.vector_stores[0]
            logger.info(f"Loading single store from {path}")
            store.load(str(path))
        else:
            # Multi-store: load each store from subdirectory
            logger.info(f"Loading {len(self.vector_stores)} stores from {path}")

            for i, (store, role) in enumerate(self.vector_stores):
                store_name = f"{store.__class__.__name__}_{role.value}_{i}"
                store_path = path / store_name

                if not store_path.exists():
                    logger.warning(f"  ⚠ Store path not found: {store_name}, skipping")
                    continue

                try:
                    store.load(str(store_path))
                    logger.info(f"  ✓ Loaded {store_name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {store_name}: {e}")
                    # Continue loading other stores

        logger.info(f"Index load completed: {path}")

    @property
    def num_chunks(self) -> int:
        """Get the number of chunks in the vector store.

        For multi-store setup, returns count from the first PRIMARY store.
        If no PRIMARY store exists, returns count from the first store.

        Returns:
            Number of chunks, or 0 if unable to determine
        """
        # Find the first PRIMARY store
        for store, role in self.vector_stores:
            if role == StorageRole.PRIMARY:
                return store.count()

        # If no PRIMARY, use the first store
        if self.vector_stores:
            return self.vector_stores[0][0].count()

        return 0
