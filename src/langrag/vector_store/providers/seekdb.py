"""SeekDB vector store implementation.

SeekDB is an AI-native database supporting vectors, full-text search,
and hybrid retrieval with built-in RRF fusion.
"""

import asyncio
from typing import Any

from loguru import logger

try:
    import pyseekdb

    SEEKDB_AVAILABLE = True
    logger.info("SeekDB available - SeekDBVectorStore ready")
except ImportError as e:
    SEEKDB_AVAILABLE = False
    logger.warning(f"pyseekdb not installed - SeekDBVectorStore unavailable: {e}")

from ...core.chunk import Chunk
from ...core.search_result import SearchResult
from ..base import BaseVectorStore
from ..capabilities import VectorStoreCapabilities


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
        host: str | None = None,
        port: int | None = None,
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
                "pyseekdb is required for SeekDBVectorStore. Install with: pip install pyseekdb"
            )

        if mode not in ("embedded", "server"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'embedded' or 'server'")

        self.collection_name = collection_name
        self.dimension = dimension
        self.mode = mode
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        # Initialize client based on mode
        if mode == "embedded":
            # Embedded mode: local database
            path = db_path
            database = collection_name  # Use collection_name as database name for simplicity

            # Use AdminClient for database management operations
            admin_client = pyseekdb.AdminClient(path=path)
            # Check if database exists using public API
            existing_dbs = [db.name for db in admin_client.list_databases()]
            if database not in existing_dbs:
                # Use public API to create database
                admin_client.create_database(database)
                logger.info(f"Created SeekDB database '{database}'")

            self._client = pyseekdb.Client(path=path, database=database)
            logger.info(f"Initialized SeekDB in embedded mode at '{path}', database '{database}'")
        else:
            # Server mode: remote SeekDB or OceanBase server
            if not host or not port:
                raise ValueError("Server mode requires host and port")
            database = collection_name

            connection_params = {
                "host": host,
                "port": int(port),
                "database": database,
                "user": "root",  # Default credentials
                "password": "",
            }

            self._client = pyseekdb.Client(**connection_params)
            logger.info(f"Initialized SeekDB in server mode: {host}:{port}, database '{database}'")

        self._collections: dict[str, Any] = {}
        self._collection_configs: dict[str, Any] = {}

        logger.info("SeekDBVectorStore ready")

        # Declare actual capabilities supported by current pyseekdb client
        self._capabilities = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=False,  # Not supported by current pyseekdb client
            supports_hybrid=False,  # Not supported by current pyseekdb client
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

    async def _get_or_create_collection(self) -> Any:
        """Get or create a collection with proper configuration."""
        if self.collection_name in self._collections:
            return self._collections[self.collection_name]

        # Check if collection exists
        exists = await asyncio.to_thread(self._client.has_collection, self.collection_name)
        if exists:
            # Collection exists, get it
            coll = await asyncio.to_thread(
                self._client.get_collection, self.collection_name, embedding_function=None
            )
            self._collections[self.collection_name] = coll
            logger.info(f"SeekDB collection '{self.collection_name}' retrieved.")
            return coll

        # Collection doesn't exist, create it
        from pyseekdb import HNSWConfiguration

        config = HNSWConfiguration(dimension=self.dimension, distance="cosine")
        self._collection_configs[self.collection_name] = config

        # Create collection without embedding function (we manage embeddings externally)
        coll = await asyncio.to_thread(
            self._client.create_collection,
            name=self.collection_name,
            configuration=config,
            embedding_function=None,  # Disable automatic embedding
        )

        self._collections[self.collection_name] = coll
        logger.info(
            f"SeekDB collection '{self.collection_name}' created with dimension={self.dimension}, distance='cosine'"
        )
        return coll

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

        # Get or create collection
        coll = await self._get_or_create_collection()

        # Prepare data for SeekDB
        ids = [chunk.id for chunk in chunks]
        embeddings_list = [chunk.embedding for chunk in chunks]

        # Include essential chunk data in metadata since SeekDB query only returns metadata
        metadatas = []
        for chunk in chunks:
            metadata = self._clean_metadata(chunk.metadata).copy()
            metadata.update(
                {
                    "content": chunk.content,
                    "source_doc_id": chunk.source_doc_id or "",
                }
            )
            metadatas.append(metadata)

        # Add to collection
        await asyncio.to_thread(coll.add, ids=ids, embeddings=embeddings_list, metadatas=metadatas)

        logger.info(f"Added {len(chunks)} embeddings to SeekDB collection '{self.collection_name}'")

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using pure vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by score descending
        """
        return asyncio.run(self._search_async(query_vector, top_k))

    async def _search_async(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """Async implementation of vector search."""
        # Get collection
        coll = await self._get_or_create_collection()

        # Perform query
        # SeekDB's query() returns: {'ids': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        results = await asyncio.to_thread(
            coll.query, query_embeddings=[query_vector], n_results=top_k
        )

        search_results = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                # Reconstruct chunk from metadata
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0

                chunk = Chunk(
                    id=doc_id,
                    content=metadata.get("content", ""),
                    embedding=None,  # Not returned in query results
                    source_doc_id=metadata.get("source_doc_id", ""),
                    metadata={
                        k: v for k, v in metadata.items() if k not in ["content", "source_doc_id"]
                    },
                )

                # Convert distance to similarity score (SeekDB returns distances, higher = less similar)
                # For cosine similarity, convert to similarity score
                score = 1.0 / (1.0 + distance) if distance > 0 else 1.0

                search_results.append(SearchResult(chunk=chunk, score=score))

        logger.debug(f"Vector search returned {len(search_results)} results")
        return search_results

    def search_fulltext(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        """Search using full-text keyword matching (BM25).

        Args:
            query_text: Query text for keyword matching
            top_k: Number of results to return

        Returns:
            List of search results, sorted by BM25 score descending
        """
        return asyncio.run(self._search_fulltext_async(query_text, top_k))

    async def _search_fulltext_async(self, query_text: str, top_k: int) -> list[SearchResult]:
        """Async implementation of full-text search."""
        # Get collection
        coll = await self._get_or_create_collection()

        # Use collection.get() with where_document filter for fulltext search
        try:
            results = await asyncio.to_thread(
                coll.get,
                where_document={"$contains": query_text},
                limit=top_k,
                include=["documents", "metadatas"],
            )

            search_results = []
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            for i, doc_id in enumerate(ids):
                # Reconstruct chunk from metadata and documents
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""

                chunk = Chunk(
                    id=doc_id,
                    content=content,
                    embedding=None,  # Not returned in fulltext results
                    source_doc_id=metadata.get("source_doc_id", ""),
                    metadata={
                        k: v for k, v in metadata.items() if k not in ["content", "source_doc_id"]
                    },
                )

                # Full-text results don't have meaningful scores yet
                # TODO: Add BM25 scores when PySeekDB exposes them
                score = 1.0  # Placeholder score

                search_results.append(SearchResult(chunk=chunk, score=score))

            logger.debug(f"Full-text search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"SeekDB fulltext search failed: {e}")
            return []

    def search_hybrid(
        self, query_vector: list[float], query_text: str, top_k: int = 5, alpha: float = 0.5
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
        return asyncio.run(self._search_hybrid_async(query_vector, query_text, top_k, alpha))

    async def _search_hybrid_async(
        self, query_vector: list[float], query_text: str, top_k: int, _alpha: float
    ) -> list[SearchResult]:
        """Async implementation of hybrid search using SeekDB's native hybrid_search."""
        # Get collection
        coll = await self._get_or_create_collection()

        # Prepare hybrid search parameters based on example code
        hybrid_args = {
            "query": {"where_document": {"$contains": query_text}, "n_results": top_k},
            "knn": {
                "query_embeddings": [query_vector],  # Single vector wrapped in list
                "n_results": top_k,
            },
            "rank": {"rrf": {}},  # Default to RRF fusion
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        try:
            results = await asyncio.to_thread(coll.hybrid_search, **hybrid_args)

            search_results = []
            # Handle the result format - results should be similar to other search methods
            if results and "ids" in results:
                # Extract data from results
                ids = results["ids"]
                metadatas = results.get("metadatas", [])
                distances = results.get("distances", [])

                # Handle nested list structure (similar to other search methods)
                if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]
                if (
                    isinstance(metadatas, list)
                    and len(metadatas) > 0
                    and isinstance(metadatas[0], list)
                ):
                    metadatas = metadatas[0]
                if (
                    isinstance(distances, list)
                    and len(distances) > 0
                    and isinstance(distances[0], list)
                ):
                    distances = distances[0]

                for i, doc_id in enumerate(ids):
                    # Reconstruct chunk from metadata
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 0.0

                    # Ensure distance is a float
                    try:
                        distance = float(distance)
                    except (TypeError, ValueError):
                        distance = 0.0

                    chunk = Chunk(
                        id=str(doc_id),
                        content=metadata.get("content", ""),
                        embedding=None,  # Not returned in hybrid results
                        source_doc_id=metadata.get("source_doc_id", ""),
                        metadata={
                            k: v
                            for k, v in metadata.items()
                            if k not in ["content", "source_doc_id"]
                        },
                    )

                    # Convert distance to similarity score
                    score = 1.0 / (1.0 + distance) if distance > 0 else 1.0

                    search_results.append(SearchResult(chunk=chunk, score=score))

            logger.debug(f"Hybrid search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"SeekDB hybrid search failed: {e}")
            # Fallback to separate searches with RRF fusion
            logger.info("Falling back to separate vector + fulltext search with RRF fusion")

            try:
                # Perform separate searches
                vector_results = await self._search_async(query_vector, top_k)
                fulltext_results = await self._search_fulltext_async(query_text, top_k)

                # Use RRF fusion from our utils
                from ...utils.rrf import reciprocal_rank_fusion

                # Convert to the format expected by RRF function
                vector_list = [(result.chunk.id, result.score) for result in vector_results]
                fulltext_list = [(result.chunk.id, result.score) for result in fulltext_results]

                # Apply RRF fusion
                fused_results = reciprocal_rank_fusion([vector_list, fulltext_list], k=top_k)

                # Reconstruct SearchResult objects
                search_results = []
                for doc_id, rrf_score in fused_results:
                    # Find the original chunk from either vector or fulltext results
                    chunk = None
                    for result in vector_results + fulltext_results:
                        if result.chunk.id == doc_id:
                            chunk = result.chunk
                            break

                    if chunk:
                        search_results.append(SearchResult(chunk=chunk, score=rrf_score))

                logger.info(f"Fallback RRF fusion completed: {len(search_results)} results")
                return search_results

            except Exception as fallback_e:
                logger.error(f"Fallback RRF fusion also failed: {fallback_e}")
                return []

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
            metadata=result.get("metadata", {}),
        )

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        asyncio.run(self._delete_async(chunk_ids))

    async def _delete_async(self, chunk_ids: list[str]):
        """Async implementation of delete."""
        # Get collection
        coll = await self._get_or_create_collection()

        # SeekDB's delete expects a where clause for filtering
        # Delete by IDs using metadata filter
        for chunk_id in chunk_ids:
            await asyncio.to_thread(coll.delete, where={"id": chunk_id})

        logger.info(
            f"Deleted {len(chunk_ids)} chunks from SeekDB collection '{self.collection_name}'"
        )

    def count(self) -> int:
        """Get the total number of chunks in this store.

        Returns:
            Number of chunks currently stored
        """
        return asyncio.run(self._count_async())

    async def _count_async(self) -> int:
        """Async implementation of count."""
        try:
            coll = await self._get_or_create_collection()
            # SeekDB collection has a count() method
            return await asyncio.to_thread(coll.count)
        except Exception as e:
            logger.error(f"Failed to count chunks in SeekDB: {e}")
            return 0

    def persist(self, _path: str) -> None:
        """Persist is handled automatically by SeekDB.

        Args:
            _path: Ignored (SeekDB manages its own persistence)
        """
        logger.info("SeekDB handles persistence automatically")

    def load(self, _path: str) -> None:
        """Load is handled automatically by SeekDB.

        Args:
            _path: Ignored (SeekDB loads on initialization)

        Raises:
            FileNotFoundError: Never raised (for interface compatibility)
        """
        logger.info("SeekDB loads data automatically on initialization")
