"""
Indexing Components - Components for document ingestion pipelines.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langrag.core.component.base import Component
from langrag.core.component.config import ChunkerConfig
from langrag.entities.document import Document
from langrag.datasource.vdb.base import BaseVector
from langrag.llm.embedder.base import BaseEmbedder
from langrag.index_processor.splitter.providers.recursive_character import RecursiveCharacterChunker
from langrag.index_processor.extractor.factory import ParserFactory

logger = logging.getLogger(__name__)


class DocumentLoaderComponent(Component):
    """
    Loads and parses documents.

    Automatically selects the appropriate parser based on file extension.

    Output keys:
        - documents: List[Document]
        - file_path: str
        - filename: str
        - doc_count: int
    """

    component_type = "document_loader"

    async def _invoke(self, file_path: str, **kwargs) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect parser
        ext = path.suffix.lower().lstrip(".")
        ext_map = {
            "md": "markdown", "markdown": "markdown",
            "htm": "html", "html": "html",
            "doc": "docx", "docx": "docx",
            "pdf": "pdf",
        }
        parser_type = ext_map.get(ext, "simple_text")

        try:
            parser = ParserFactory.create(parser_type)
        except ValueError:
            logger.warning(f"Parser for '{ext}' not found, using simple_text")
            parser = ParserFactory.create("simple_text")

        documents = parser.parse(path)
        logger.info(f"[DocumentLoader] Parsed {len(documents)} documents from {path.name}")

        return {
            "documents": documents,
            "file_path": str(path),
            "filename": path.name,
            "doc_count": len(documents),
        }


class ChunkingComponent(Component):
    """
    Splits documents into smaller chunks.

    Input keys:
        - documents: List[Document]

    Output keys:
        - chunks: List[Document]
        - chunk_count: int
    """

    component_type = "chunker"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        config: Optional[ChunkerConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if config:
            self.chunk_size = config.chunk_size
            self.chunk_overlap = config.chunk_overlap
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

    async def _invoke(self, documents: List[Document], **kwargs) -> Dict[str, Any]:
        chunker = RecursiveCharacterChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = chunker.split(documents)
        logger.info(f"[Chunker] Created {len(chunks)} chunks")

        return {"chunks": chunks, "chunk_count": len(chunks)}


class EmbeddingComponent(Component):
    """
    Generates embeddings for document chunks.

    The embedder can be provided at init time or passed at runtime.
    Runtime embedder takes precedence.

    Input keys:
        - chunks: List[Document]
        - embedder (optional): BaseEmbedder - runtime override

    Output keys:
        - chunks: List[Document] (with vectors attached)
        - embedded_count: int
    """

    component_type = "embedder"

    def __init__(self, embedder: Optional[BaseEmbedder] = None, **kwargs):
        super().__init__(**kwargs)
        self._default_embedder = embedder

    async def _invoke(
        self,
        chunks: List[Document],
        embedder: Optional[BaseEmbedder] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Runtime embedder takes precedence
        active_embedder = embedder or self._default_embedder

        if not chunks:
            return {"chunks": [], "embedded_count": 0}

        if not active_embedder:
            logger.warning("[Embedder] No embedder provided, skipping embedding")
            return {"chunks": chunks, "embedded_count": 0}

        texts = [c.page_content for c in chunks]

        import asyncio
        vectors = await asyncio.to_thread(active_embedder.embed, texts)

        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector

        logger.info(f"[Embedder] Embedded {len(chunks)} chunks")
        return {"chunks": chunks, "embedded_count": len(chunks)}


class VectorStoreComponent(Component):
    """
    Stores document chunks in a vector database.

    The vector store can be provided at init time or passed at runtime.

    Input keys:
        - chunks: List[Document]
        - vector_store (optional): BaseVector - runtime override

    Output keys:
        - stored_count: int
        - collection_name: str
    """

    component_type = "vector_store"

    def __init__(self, vector_store: Optional[BaseVector] = None, **kwargs):
        super().__init__(**kwargs)
        self._default_store = vector_store

    async def _invoke(
        self,
        chunks: List[Document],
        vector_store: Optional[BaseVector] = None,
        **kwargs
    ) -> Dict[str, Any]:
        active_store = vector_store or self._default_store

        if not active_store:
            raise ValueError("No vector store provided")

        if not chunks:
            return {"stored_count": 0, "collection_name": active_store.collection_name}

        import asyncio
        await asyncio.to_thread(active_store.add_texts, chunks)

        logger.info(f"[VectorStore] Stored {len(chunks)} chunks")
        return {
            "stored_count": len(chunks),
            "collection_name": active_store.collection_name,
        }
