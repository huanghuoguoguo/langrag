"""
Ingestion Pipeline Module.

This pipeline handles the standard document indexing flow:
Load -> Parse -> Chunk -> Embed -> Store

It supports various techniques:
- High Quality (Standard)
- QA (LLM-generated Q&A)
- Parent-Child (Hierarchical)
"""

import logging
from pathlib import Path
from typing import Any

from langrag import (
    BaseEmbedder,
    BaseVector,
    ParentChildIndexProcessor,
    QAIndexProcessor,
    RecursiveCharacterChunker,
    SimpleTextParser,
)
from langrag.entities.document import Document
from langrag.datasource.kv.base import BaseKVStore
from langrag.llm.base import BaseLLM
from langrag.index_processor.extractor.factory import ParserFactory
from langrag.index_processor.processor.page_index import PageIndexProcessor
from langrag.pipeline.base import BasePipeline

logger = logging.getLogger(__name__)

class IngestionPipeline(BasePipeline):
    """
    Standard ingestion pipeline for LangRAG.
    
    Orchestrates:
    1. Automatic parser selection based on file extension
    2. Text chunking
    3. Vector embedding
    4. Storage in Vector DB and optional KV Store
    """

    def __init__(
        self,
        vector_store: BaseVector,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
        kv_store: BaseKVStore | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            vector_store: Target vector store for storing chunks/embeddings.
            embedder: Embedding model (required for vector search).
            llm: LLM instance (required for QA indexing).
            kv_store: Key-Value store (required for Parent-Child indexing).
            chunk_size: Default chunk size.
            chunk_overlap: Default chunk overlap.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.kv_store = kv_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(
        self, 
        file_path: Path | str, 
        indexing_technique: str = "high_quality",
        **kwargs
    ) -> int:
        """
        Run the ingestion pipeline on a single file.

        Args:
            file_path: Path to the file to ingest.
            indexing_technique: "high_quality", "qa", or "parent_child".
            **kwargs: Override config (e.g., chunk_size).

        Returns:
            Number of indexed items (chunks).
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        logger.info(f"[Ingestion] Processing {file_path}, technique={indexing_technique}")

        # 1. Parse
        parser = self._get_parser(file_path)
        raw_docs = parser.parse(file_path)
        
        if not raw_docs:
            logger.warning(f"No content parsed from {file_path}")
            return 0

        # 2. Process based on technique
        if indexing_technique == "qa":
            return self._process_qa(raw_docs, chunk_size, chunk_overlap)
        
        elif indexing_technique == "parent_child":
            return self._process_parent_child(raw_docs, chunk_size, chunk_overlap)
            
        elif indexing_technique == "page_index":
            return self._process_page_index(raw_docs)
        
        else:
            return self._process_high_quality(raw_docs, chunk_size, chunk_overlap)

    def _get_parser(self, file_path: Path) -> Any:
        """Get appropriate parser from factory based on extension."""
        ext = file_path.suffix.lower().lstrip(".")
        if not ext:
            ext = "simple_text"
            
        # Standardize extensions for factory
        if ext in ["md", "markdown"]:
            ext = "markdown"
        elif ext in ["htm", "html"]:
            ext = "html"
        elif ext in ["doc", "docx"]:
            ext = "docx"
        elif ext == "pdf":
            ext = "pdf"
        else:
            # Fallback for unknown types if not explicitly handled
            # But let's see if factory supports 'txt'
            if ext not in ParserFactory.list_types() and ext not in ["txt", "text"]:
                ext = "simple_text"
        
        try:
            return ParserFactory.create(ext)
        except ValueError:
            logger.warning(f"Parser for '{ext}' not found/installed, falling back to simple_text")
            return SimpleTextParser()

    def _process_high_quality(
        self, 
        docs: list[Document], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> int:
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = chunker.split(docs)

        if self.embedder:
            logger.info("Generating embeddings...")
            texts = [c.page_content for c in chunks]
            vectors = self.embedder.embed(texts)
            for doc, vec in zip(chunks, vectors):
                doc.vector = vec

        logger.info(f"Storing {len(chunks)} chunks...")
        self.vector_store.add_texts(chunks)
        return len(chunks)

    def _process_qa(
        self, 
        docs: list[Document], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> int:
        if not self.llm:
            raise ValueError("LLM required for QA indexing")
        if not self.embedder:
            raise ValueError("Embedder required for QA indexing")

        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        processor = QAIndexProcessor(
            vector_store=self.vector_store,
            llm=self.llm,
            embedder=self.embedder,
            splitter=chunker
        )
        processor.process(self.vector_store.dataset, docs)
        return len(docs) * 2 # Estimation

    def _process_parent_child(
        self, 
        docs: list[Document], 
        chunk_size: int, 
        chunk_overlap: int
    ) -> int:
        if not self.kv_store:
            raise ValueError("KV Store required for Parent-Child indexing")
        if not self.embedder:
            raise ValueError("Embedder required for Parent-Child indexing")

        parent_chunk_size = chunk_size * 2
        child_chunk_size = max(chunk_size // 2, 200)

        parent_splitter = RecursiveCharacterChunker(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
        child_splitter = RecursiveCharacterChunker(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap // 2)

        processor = ParentChildIndexProcessor(
            vector_store=self.vector_store,
            kv_store=self.kv_store,
            embedder=self.embedder,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        processor.process(self.vector_store.dataset, docs)
        return len(docs) * 4 # Estimation

    def _process_page_index(
        self,
        docs: list[Document]
    ) -> int:
        if not self.llm:
            raise ValueError("LLM required for PageIndex")
        
        # Embedder is optional in PageIndexProcessor (can use LLM), 
        # but Pipeline usually has it.
        
        processor = PageIndexProcessor(
            llm=self.llm,
            embedder=self.embedder,
            vector_manager=self.vector_store
        )
        
        processor.process(self.vector_store.dataset, docs)
        return len(docs) * 5 # Estimation (Tree nodes + summaries)
