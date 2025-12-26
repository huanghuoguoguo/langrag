"""SeekDB vector store implementation.

SeekDB is an AI-native database supporting vectors, full-text search,
and hybrid retrieval with built-in RRF fusion.
"""

import asyncio
import re
from typing import Optional, Any
from loguru import logger

try:
    import pyseekdb
    PySeekDB = pyseekdb.Client  # pyseekdb uses Client class
    SEEKDB_AVAILABLE = True
    logger.info("SeekDB available - SeekDBVectorStore ready")
except ImportError as e:
    SEEKDB_AVAILABLE = False
    logger.warning(f"pyseekdb not installed - SeekDBVectorStore unavailable: {e}")

from ..base import BaseVectorStore
from ..capabilities import VectorStoreCapabilities
from ...core.chunk import Chunk
from ...core.search_result import SearchResult


class SeekDBVectorStore(BaseVectorStore):
    """SeekDB vector store with full-text and hybrid search support.

    SeekDB provides native support for:
    - Vector similarity search (HNSW indexing)
    - Full-text keyword search (BM25)
    - Hybrid search (vector + text with internal RRF fusion)

    Attributes:
        collection_name: Name of the SeekDB collection
        dimension: Embedding dimension
        mode: "embedded" or "server"
        _client: SeekDB client instance
        _capabilities: Full support for all search modes
    """

    def __init__(
        self,
        collection_name: str = "langrag_chunks",
        dimension: int = 384,
        mode: str = "embedded",
        db_path: str = "./seekdb_data",
        host: Optional[str] = None,
        port: Optional[int] = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        """Initialize SeekDB vector store.

        Args:
            collection_name: Name of the collection to use
            dimension: Embedding vector dimension
            mode: "embedded" or "server"
            db_path: Database path for embedded mode
            host: Server host for server mode
            port: Server port for server mode
            hnsw_m: HNSW M parameter (neighbors per layer)
            hnsw_ef_construction: HNSW ef_construction parameter
            hnsw_ef_search: HNSW ef_search parameter

        Raises:
            ImportError: If pyseekdb is not installed
            ValueError: If mode is invalid
        """
        if not SEEKDB_AVAILABLE:
            raise ImportError(
                "pyseekdb is required for SeekDBVectorStore. "
                "Install with: pip install pyseekdb"
            )

        if mode not in ("embedded", "server"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'embedded' or 'server'")

        self.collection_name = collection_name
        self.dimension = dimension
        self.mode = mode
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        # Initialize client
        if mode == "embedded":
            # For embedded mode, just use the path
            self._client = PySeekDB(path=db_path)
            logger.info(f"Initialized SeekDB in embedded mode: {db_path}")
        else:
            if not host or not port:
                raise ValueError("Server mode requires host and port")
            # For server mode - this needs to be verified with actual API
            self._client = PySeekDB(host=host, port=port)
            logger.info(f"Initialized SeekDB in server mode: {host}:{port}")

        # Create collection if needed
        # Temporarily disabled for API compatibility testing
        # asyncio.run(self._ensure_collection())
        logger.info("SeekDB client initialized (collection management deferred)")

        # Declare full capabilities
        self._capabilities = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=True,
            supports_hybrid=True
        )

        logger.info(
            f"SeekDBVectorStore ready: collection={collection_name}, "
            f"dimension={dimension}, HNSW(M={hnsw_m}, ef_c={hnsw_ef_construction})"
        )

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        """Get capabilities - full support for all search modes.

        Returns:
            Capability declaration with all modes enabled
        """
        return self._capabilities

    async def _ensure_collection(self):
        """Ensure collection exists with proper schema."""
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self._client.list_collections)

            if self.collection_name not in collections:
                # Create collection with vector and text fields
                schema = {
                    "id": "string",
                    "content": "text",  # Full-text indexed
                    "embedding": f"vector({self.dimension})",
                    "source_doc_id": "string",
                    "metadata": "json"
                }

                await asyncio.to_thread(
                    self._client.create_collection,
                    name=self.collection_name,
                    schema=schema,
                    vector_index_config={
                        "type": "hnsw",
                        "m": self._hnsw_m,
                        "ef_construction": self._hnsw_ef_construction,
                    }
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata for SeekDB storage.

        Escapes special characters and removes problematic values.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Cleaned metadata safe for SeekDB
        """
        cleaned = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert to string and escape special chars if needed
            if isinstance(value, str):
                # Escape backslashes and quotes
                value = value.replace("\\", "\\\\").replace('"', '\\"')

            cleaned[key] = value

        return cleaned

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to SeekDB.

        Args:
            chunks: Chunks with embeddings to store

        Raises:
            ValueError: If any chunk lacks an embedding
        """
        asyncio.run(self._add_async(chunks))

    async def _add_async(self, chunks: list[Chunk]):
        """Async implementation of add."""
        if not chunks:
            return

        # Validate embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} missing embedding")

        # Prepare documents
        documents = []
        for chunk in chunks:
            doc = {
                "id": chunk.id,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "source_doc_id": chunk.source_doc_id or "",
                "metadata": self._clean_metadata(chunk.metadata)
            }
            documents.append(doc)

        # Batch insert
        await asyncio.to_thread(
            self._client.insert,
            collection=self.collection_name,
            documents=documents
        )

        logger.info(f"Added {len(chunks)} chunks to SeekDB collection {self.collection_name}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[SearchResult]:
        """Search using pure vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending
        """
        return asyncio.run(self._search_async(query_vector, top_k))

    async def _search_async(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[SearchResult]:
        """Async implementation of vector search."""
        results = await asyncio.to_thread(
            self._client.search,
            collection=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            ef_search=self._hnsw_ef_search
        )

        search_results = []
        for result in results:
            chunk = self._result_to_chunk(result)
            score = float(result.get("score", 0.0))
            search_results.append(SearchResult(chunk=chunk, score=score))

        logger.debug(f"Vector search returned {len(search_results)} results")
        return search_results

    def search_fulltext(
        self,
        query_text: str,
        top_k: int = 5
    ) -> list[SearchResult]:
        """Search using full-text keyword matching (BM25).

        Args:
            query_text: Query text for keyword matching
            top_k: Number of results to return

        Returns:
            List of search results, sorted by BM25 score descending
        """
        return asyncio.run(self._search_fulltext_async(query_text, top_k))

    async def _search_fulltext_async(
        self,
        query_text: str,
        top_k: int
    ) -> list[SearchResult]:
        """Async implementation of full-text search."""
        results = await asyncio.to_thread(
            self._client.search_text,
            collection=self.collection_name,
            query=query_text,
            top_k=top_k
        )

        search_results = []
        for result in results:
            chunk = self._result_to_chunk(result)
            score = float(result.get("score", 0.0))
            search_results.append(SearchResult(chunk=chunk, score=score))

        logger.debug(f"Full-text search returned {len(search_results)} results")
        return search_results

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> list[SearchResult]:
        """Search using native hybrid mode (vector + text with RRF).

        SeekDB performs internal RRF fusion of vector and text results.

        Args:
            query_vector: Query embedding vector
            query_text: Query text for keyword matching
            top_k: Number of results to return
            alpha: Weight for vector vs text (0.0=text only, 1.0=vector only)

        Returns:
            List of search results with combined scores
        """
        return asyncio.run(
            self._search_hybrid_async(query_vector, query_text, top_k, alpha)
        )

    async def _search_hybrid_async(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        alpha: float
    ) -> list[SearchResult]:
        """Async implementation of hybrid search."""
        results = await asyncio.to_thread(
            self._client.search_hybrid,
            collection=self.collection_name,
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            alpha=alpha,
            ef_search=self._hnsw_ef_search
        )

        search_results = []
        for result in results:
            chunk = self._result_to_chunk(result)
            score = float(result.get("score", 0.0))
            search_results.append(SearchResult(chunk=chunk, score=score))

        logger.debug(f"Hybrid search returned {len(search_results)} results")
        return search_results

    def _result_to_chunk(self, result: dict) -> Chunk:
        """Convert SeekDB result to Chunk object.

        Args:
            result: Result dictionary from SeekDB

        Returns:
            Chunk instance
        """
        return Chunk(
            id=result["id"],
            content=result["content"],
            embedding=result.get("embedding"),
            source_doc_id=result.get("source_doc_id", ""),
            metadata=result.get("metadata", {})
        )

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        asyncio.run(self._delete_async(chunk_ids))

    async def _delete_async(self, chunk_ids: list[str]):
        """Async implementation of delete."""
        await asyncio.to_thread(
            self._client.delete,
            collection=self.collection_name,
            ids=chunk_ids
        )
        logger.info(f"Deleted {len(chunk_ids)} chunks from SeekDB")

    def persist(self, path: str) -> None:
        """Persist is handled automatically by SeekDB.

        Args:
            path: Ignored (SeekDB manages its own persistence)
        """
        logger.info("SeekDB handles persistence automatically")

    def load(self, path: str) -> None:
        """Load is handled automatically by SeekDB.

        Args:
            path: Ignored (SeekDB loads on initialization)

        Raises:
            FileNotFoundError: Never raised (for interface compatibility)
        """
        logger.info("SeekDB loads data automatically on initialization")
