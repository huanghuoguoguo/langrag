"""Fixed-size chunker implementation."""

from loguru import logger

from ...core.chunk import Chunk
from ...core.document import Document
from ..base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size segments with optional overlap.

    This chunker splits text based on character count with a sliding
    window approach to preserve context across chunk boundaries.

    Attributes:
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between consecutive chunks
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks

        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into fixed-size chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with metadata from source documents
        """
        chunks = []

        for doc in documents:
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)
            logger.debug(f"Split document {doc.id} into {len(doc_chunks)} chunks")

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _split_document(self, doc: Document) -> list[Chunk]:
        """Split a single document into chunks.

        Args:
            doc: Document to split

        Returns:
            List of chunks from this document
        """
        text = doc.content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk = Chunk(
                content=chunk_text,
                source_doc_id=doc.id,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": min(end, len(text)),
                },
            )
            chunks.append(chunk)

            start += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks
