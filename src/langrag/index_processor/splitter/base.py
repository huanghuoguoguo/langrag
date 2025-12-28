"""Base chunker interface."""

from abc import ABC, abstractmethod

from ..core.chunk import Chunk
from ..core.document import Document


class BaseChunker(ABC):
    """Abstract base class for text chunking.

    Chunkers split documents into smaller pieces suitable
    for embedding and retrieval.
    """

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with metadata preserved from source documents
        """
        pass
