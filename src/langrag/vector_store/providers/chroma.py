"""Chroma vector store implementation.

ChromaDB is a popular open-source embedding database focused on
vector similarity search. It does not support full-text or hybrid
search natively, but can be combined with other stores using RRF.
"""

from typing import Optional, Any, Dict
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB available - ChromaVectorStore ready")
except ImportError as e:
    CHROMA_AVAILABLE = False
    logger.warning(f"chromadb not installed - ChromaVectorStore unavailable: {e}")

from ..base import BaseVectorStore
from ..capabilities import VectorStoreCapabilities
from ...core.chunk import Chunk
from ...core.search_result import SearchResult


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store with pure vector similarity search.

    ChromaDB provides efficient vector similarity search with metadata
    filtering. It supports both persistent (on-disk) and ephemeral
    (in-memory) modes.

    Capabilities:
    - ✅ Vector similarity search
    - ❌ Full-text search (not supported)
    - ❌ Native hybrid search (use RRF with text store instead)

    Attributes:
        collection_name: Name of the Chroma collection
        persist_directory: Directory for persistent storage (None for ephemeral)
        distance_metric: Distance function (l2, ip, or cosine)
        _client: ChromaDB client instance
        _collection: ChromaDB collection instance
        _capabilities: Declares vector-only support
    """

    def __init__(
        self,
        collection_name: str = "langrag_chunks",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
        client_settings: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory for persistent storage (None = ephemeral)
            distance_metric: Distance function - "l2", "ip", or "cosine"
            client_settings: Additional Chroma client settings

        Raises:
            ImportError: If chromadb is not installed
            ValueError: If distance_metric is invalid
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install with: pip install chromadb"
            )

        valid_metrics = {"l2", "ip", "cosine"}
        if distance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid distance_metric: '{distance_metric}'. "
                f"Must be one of: {', '.join(valid_metrics)}"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # Initialize client
        if persist_directory:
            # Persistent mode - use PersistentClient
            self._client = chromadb.PersistentClient(path=persist_directory, settings=client_settings)
            logger.info(f"Initialized Chroma in persistent mode: {persist_directory}")
        else:
            # Ephemeral mode (in-memory) - use Client
            settings = Settings(
                anonymized_telemetry=False,
                **(client_settings or {})
            )
            self._client = chromadb.Client(settings)
            logger.info("Initialized Chroma in ephemeral mode (in-memory)")

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

        logger.info(
            f"ChromaVectorStore ready: collection={collection_name}, "
            f"metric={distance_metric}, "
            f"items={self._collection.count()}"
        )

        # Declare capabilities (vector only, no full-text or hybrid)
        self._capabilities = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=False,
            supports_hybrid=False
        )

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        """Get capabilities - vector search only.

        Returns:
            Capability declaration indicating vector-only support
        """
        return self._capabilities

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to Chroma collection.

        Args:
            chunks: Chunks with embeddings to store

        Raises:
            ValueError: If any chunk lacks an embedding
        """
        if not chunks:
            return

        # Validate embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} missing embedding")

        # Prepare data for Chroma
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]

        # Prepare metadata (Chroma requires all values to be basic types)
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source_doc_id": chunk.source_doc_id or "",
                # Convert all metadata values to strings for Chroma compatibility
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            metadatas.append(metadata)

        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(
            f"Added {len(chunks)} chunks to Chroma collection {self.collection_name} "
            f"(total: {self._collection.count()})"
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> list[SearchResult]:
        """Search using vector similarity.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            metadata_filter: Optional Chroma where filter (e.g., {"source": "doc1"})

        Returns:
            List of search results, sorted by similarity descending
        """
        # Query Chroma
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to SearchResult objects
        search_results = []

        if not results['ids'] or not results['ids'][0]:
            logger.debug("No results found")
            return []

        # Chroma returns nested lists (one per query)
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i, chunk_id in enumerate(ids):
            # Extract metadata
            metadata = metadatas[i] if i < len(metadatas) else {}
            source_doc_id = metadata.pop("source_doc_id", "")

            # Create Chunk (embeddings not needed in search results)
            chunk = Chunk(
                id=chunk_id,
                content=documents[i],
                embedding=None,  # Not returned in search results
                source_doc_id=source_doc_id,
                metadata=metadata
            )

            # Convert distance to similarity score [0, 1]
            # For cosine: distance is already 1 - similarity, so invert it
            # For L2: use exponential decay
            # For IP: distance is negative dot product
            distance = distances[i]
            if self.distance_metric == "cosine":
                # Cosine distance: range [-1, 1] → similarity [0, 1]
                # distance = 1 - cosine_similarity, so cosine_similarity = 1 - distance
                # But cosine similarity can be negative, so clamp it
                cosine_sim = 1.0 - distance
                score = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))  # Normalize to [0, 1]
            elif self.distance_metric == "l2":
                # L2 distance: always >= 0, convert to similarity
                score = 1.0 / (1.0 + distance)  # Range: (0, 1]
                score = max(0.0, min(1.0, score))  # Ensure [0, 1]
            else:  # ip (inner product)
                # IP distance can be negative, normalize to [0, 1]
                # Simple approach: use sigmoid-like function
                score = max(0.0, min(1.0, 1.0 / (1.0 + abs(distance))))

            search_results.append(SearchResult(chunk=chunk, score=score))

        logger.debug(
            f"Chroma search returned {len(search_results)} results "
            f"(requested top_k={top_k})"
        )

        return search_results

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        if not chunk_ids:
            return

        self._collection.delete(ids=chunk_ids)
        logger.info(
            f"Deleted {len(chunk_ids)} chunks from Chroma collection "
            f"(remaining: {self._collection.count()})"
        )

    def count(self) -> int:
        """Get the total number of chunks in this store.

        Returns:
            Number of chunks currently stored
        """
        return self._collection.count()

    def persist(self, path: str) -> None:
        """Persist is automatic in Chroma persistent mode.

        Args:
            path: Ignored (Chroma manages persistence automatically)
        """
        if self.persist_directory:
            # In persistent mode, data is automatically saved
            logger.info(
                f"Chroma collection {self.collection_name} persisted automatically "
                f"to {self.persist_directory}"
            )
        else:
            logger.warning(
                f"Collection {self.collection_name} is ephemeral (in-memory). "
                "Data will be lost on exit. Use persist_directory for persistence."
            )

    def load(self, path: str) -> None:
        """Load is automatic in Chroma persistent mode.

        Args:
            path: Ignored (Chroma loads on initialization)

        Raises:
            FileNotFoundError: Never raised (for interface compatibility)
        """
        if self.persist_directory:
            logger.info(
                f"Chroma collection {self.collection_name} loaded from "
                f"{self.persist_directory} ({self._collection.count()} chunks)"
            )
        else:
            logger.info(
                f"Collection {self.collection_name} is ephemeral. "
                "No data to load."
            )

    def clear(self) -> None:
        """Clear all chunks from the collection.

        Useful for testing or resetting the store.
        """
        # Get all IDs
        all_results = self._collection.get()
        if all_results['ids']:
            self._collection.delete(ids=all_results['ids'])
            logger.info(f"Cleared all chunks from collection {self.collection_name}")
        else:
            logger.debug(f"Collection {self.collection_name} already empty")

    def count(self) -> int:
        """Get the number of chunks in the collection.

        Returns:
            Number of chunks stored
        """
        return self._collection.count()
