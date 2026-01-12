"""
Document processing service for the Web layer.

This module acts as a facade around the standard LangRAG IngestionPipeline.
It delegates all heavy lifting to the core library.
"""

import logging
from pathlib import Path

from langrag import BaseEmbedder, BaseVector
from langrag.datasource.kv.base import BaseKVStore
from langrag.llm.base import BaseLLM
from langrag.pipeline.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Facade for the standard LangRAG IngestionPipeline.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        llm_adapter: BaseLLM | None = None,
        kv_store: BaseKVStore | None = None
    ):
        self.embedder = embedder
        self.llm_adapter = llm_adapter
        self.kv_store = kv_store

    def process(
        self,
        file_path: Path,
        vector_store: BaseVector,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        indexing_technique: str = "high_quality"
    ) -> int:
        """
        Process a document using the IngestionPipeline.
        """
        pipeline = IngestionPipeline(
            vector_store=vector_store,
            embedder=self.embedder,
            llm=self.llm_adapter,
            kv_store=self.kv_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return pipeline.run(
            file_path=file_path,
            indexing_technique=indexing_technique
        )
