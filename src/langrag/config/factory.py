"""Component factory for dynamic loading using type-based configuration."""

from typing import Any

from loguru import logger

from langrag.datasource.vdb.factory import VectorStoreFactory
from langrag.index_processor.extractor import BaseParser, ParserFactory
from langrag.index_processor.splitter import BaseChunker, ChunkerFactory
from langrag.llm import BaseLLM, LLMFactory
from langrag.llm.embedder import BaseEmbedder, EmbedderFactory
from langrag.retrieval.compressor import BaseCompressor, CompressorFactory
from langrag.retrieval.rerank import BaseReranker, RerankerFactory

from .models import ComponentConfig


class ComponentFactory:
    """Unified factory for creating all component types.

    This factory delegates to specialized factories based on component type.
    """

    @staticmethod
    def create_parser(config: ComponentConfig) -> BaseParser:
        """Create a parser from configuration."""
        logger.info(f"Creating parser: {config.type}")
        return ParserFactory.create(config.type, **config.params)

    @staticmethod
    def create_chunker(config: ComponentConfig) -> BaseChunker:
        """Create a chunker from configuration."""
        logger.info(f"Creating chunker: {config.type}")
        return ChunkerFactory.create(config.type, **config.params)

    @staticmethod
    def create_embedder(config: ComponentConfig) -> BaseEmbedder:
        """Create an embedder from configuration."""
        logger.info(f"Creating embedder: {config.type}")
        return EmbedderFactory.create(config.type, **config.params)

    @staticmethod
    def create_vector_store(config: ComponentConfig, dataset: Any) -> Any:
        """Create a vector store from configuration.

        Args:
            config: Component configuration
            dataset: The dataset instance (Dataset type, typed as Any to avoid circular imports if necessary)

        Returns:
            BaseVector instance
        """
        logger.info(f"Creating vector store: {config.type}")
        return VectorStoreFactory.create(config.type, dataset, **config.params)

    @staticmethod
    def create_reranker(config: ComponentConfig) -> BaseReranker:
        """Create a reranker from configuration."""
        logger.info(f"Creating reranker: {config.type}")
        return RerankerFactory.create(config.type, **config.params)

    @staticmethod
    def create_compressor(config: ComponentConfig) -> BaseCompressor:
        """Create a compressor from configuration."""
        logger.info(f"Creating compressor: {config.type}")
        return CompressorFactory.create(config.type, **config.params)

    @staticmethod
    def create_llm(config: ComponentConfig) -> BaseLLM:
        """Create an LLM from configuration."""
        logger.info(f"Creating LLM: {config.type}")
        return LLMFactory.create(config.type, **config.params)
