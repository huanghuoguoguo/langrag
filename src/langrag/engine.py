"""RAG Engine - High-level orchestrator for RAG operations."""

from pathlib import Path
from loguru import logger

from .config.models import RAGConfig
from .config.factory import ComponentFactory
from .pipeline.indexing import IndexingPipeline
from .pipeline.retrieval import RetrievalPipeline
from .core.search_result import SearchResult


class RAGEngine:
    """High-level orchestrator for RAG operations.

    This class manages the complete RAG system lifecycle, including:
    - Component initialization from configuration
    - Pipeline creation and management
    - High-level indexing and retrieval operations

    Attributes:
        config: RAG system configuration
        parser: Document parser
        chunker: Text chunker
        embedder: Embedding generator
        vector_store: Vector storage backend
        reranker: Optional reranker
        llm: Optional LLM for generation
        indexing_pipeline: Pipeline for document indexing
        retrieval_pipeline: Pipeline for query retrieval
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
        self.vector_store = ComponentFactory.create_vector_store(
            self.config.vector_store
        )

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

        self.indexing_pipeline = IndexingPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        self.retrieval_pipeline = RetrievalPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
            reranker=self.reranker,
            top_k=self.config.retrieval_top_k,
            rerank_top_k=self.config.rerank_top_k,
        )

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
        """
        if hasattr(self.vector_store, "_chunks"):
            return len(self.vector_store._chunks)
        return 0
