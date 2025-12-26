"""In-memory vector store implementation."""

import pickle
from pathlib import Path
from loguru import logger

from ..base import BaseVectorStore
from ..capabilities import VectorStoreCapabilities
from ...core.chunk import Chunk
from ...core.search_result import SearchResult
from ...utils.similarity import cosine_similarity


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store using cosine similarity.

    WARNING: This implementation uses pickle for persistence, which
    has security implications. Only load from trusted sources.

    Attributes:
        _chunks: Dictionary mapping chunk IDs to Chunk objects
        _capabilities: Declares this store only supports vector search
    """

    def __init__(self):
        """Initialize an empty vector store."""
        self._chunks: dict[str, Chunk] = {}
        self._capabilities = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=False,
            supports_hybrid=False
        )
        logger.info("Initialized InMemoryVectorStore")

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        """Get capabilities - only vector search supported.

        Returns:
            Capability declaration indicating vector-only support
        """
        return self._capabilities

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store.

        Args:
            chunks: Chunks with embeddings to store

        Raises:
            ValueError: If any chunk lacks an embedding
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} missing embedding")
            self._chunks[chunk.id] = chunk

        logger.info(f"Added {len(chunks)} chunks to store (total: {len(self._chunks)})")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[SearchResult]:
        """Search using cosine similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending
        """
        if not self._chunks:
            logger.warning("Store is empty, returning no results")
            return []

        results = []
        for chunk in self._chunks.values():
            score = cosine_similarity(query_vector, chunk.embedding)
            results.append(SearchResult(chunk=chunk, score=score))

        # Sort by score descending
        results.sort(reverse=True)

        logger.debug(
            f"Found {len(results)} results, returning top {top_k}"
        )
        return results[:top_k]

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
        logger.info(f"Deleted {len(chunk_ids)} chunks")

    def persist(self, path: str) -> None:
        """Save store to pickle file.

        WARNING: Only load from trusted sources.

        Args:
            path: File path to save to
        """
        Path(path).write_bytes(pickle.dumps(self._chunks))
        logger.info(f"Persisted store to {path} ({len(self._chunks)} chunks)")

    def load(self, path: str) -> None:
        """Load store from pickle file.

        WARNING: Only load from trusted sources.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

        self._chunks = pickle.loads(Path(path).read_bytes())
        logger.info(f"Loaded {len(self._chunks)} chunks from {path}")
