"""Component factory for dynamic loading using type-based configuration."""

from loguru import logger

from .models import ComponentConfig
from ..parser import ParserFactory, BaseParser
from ..chunker import ChunkerFactory, BaseChunker
from ..embedder import EmbedderFactory, BaseEmbedder
from ..vector_store import VectorStoreFactory, BaseVectorStore
from ..reranker import RerankerFactory, BaseReranker
from ..llm import LLMFactory, BaseLLM


class ComponentFactory:
    """Unified factory for creating all component types.

    This factory delegates to specialized factories based on component type.
    """

    @staticmethod
    def create_parser(config: ComponentConfig) -> BaseParser:
        """Create a parser from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            Parser instance
        """
        logger.info(f"Creating parser: {config.type}")
        return ParserFactory.create(config.type, **config.params)

    @staticmethod
    def create_chunker(config: ComponentConfig) -> BaseChunker:
        """Create a chunker from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            Chunker instance
        """
        logger.info(f"Creating chunker: {config.type}")
        return ChunkerFactory.create(config.type, **config.params)

    @staticmethod
    def create_embedder(config: ComponentConfig) -> BaseEmbedder:
        """Create an embedder from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            Embedder instance
        """
        logger.info(f"Creating embedder: {config.type}")
        return EmbedderFactory.create(config.type, **config.params)

    @staticmethod
    def create_vector_store(config: ComponentConfig) -> BaseVectorStore:
        """Create a vector store from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            Vector store instance
        """
        logger.info(f"Creating vector store: {config.type}")
        return VectorStoreFactory.create(config.type, **config.params)

    @staticmethod
    def create_reranker(config: ComponentConfig) -> BaseReranker:
        """Create a reranker from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            Reranker instance
        """
        logger.info(f"Creating reranker: {config.type}")
        return RerankerFactory.create(config.type, **config.params)

    @staticmethod
    def create_llm(config: ComponentConfig) -> BaseLLM:
        """Create an LLM from configuration.

        Args:
            config: Component configuration with type and params

        Returns:
            LLM instance
        """
        logger.info(f"Creating LLM: {config.type}")
        return LLMFactory.create(config.type, **config.params)
