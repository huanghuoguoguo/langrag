"""Indexing pipeline for document processing."""

from pathlib import Path
from loguru import logger

from ..parser import BaseParser
from ..chunker import BaseChunker
from ..embedder import BaseEmbedder
from ..vector_store import BaseVectorStore


class IndexingPipeline:
    """Pipeline for indexing documents into a vector store.

    This pipeline orchestrates the complete indexing workflow:
    1. Parse files into documents
    2. Chunk documents into smaller pieces
    3. Generate embeddings for chunks
    4. Store chunks with embeddings in vector store

    Attributes:
        parser: Document parser
        chunker: Text chunker
        embedder: Embedding generator
        vector_store: Vector storage backend
    """

    def __init__(
        self,
        parser: BaseParser,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ):
        """Initialize the indexing pipeline.

        Args:
            parser: Parser for reading documents
            chunker: Chunker for splitting text
            embedder: Embedder for generating vectors
            vector_store: Store for persisting chunks
        """
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index_file(self, file_path: str | Path) -> int:
        """Index a single file.

        Args:
            file_path: Path to file to index

        Returns:
            Number of chunks indexed

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or chunks lack embeddings
        """
        logger.info(f"Indexing file: {file_path}")

        # 1. Parse
        documents = self.parser.parse(file_path)
        logger.debug(f"Parsed {len(documents)} documents")

        # 2. Chunk
        chunks = self.chunker.split(documents)
        logger.debug(f"Created {len(chunks)} chunks")

        # 3. Embed
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # 4. Store
        self.vector_store.add(chunks)

        logger.info(f"Successfully indexed {len(chunks)} chunks")
        return len(chunks)

    def index_files(self, file_paths: list[str | Path]) -> int:
        """Index multiple files.

        Args:
            file_paths: List of file paths to index

        Returns:
            Total number of chunks indexed
        """
        total = 0
        for path in file_paths:
            total += self.index_file(path)
        return total
