"""DuckDB vector store implementation.

DuckDB supports both vector similarity search (VSS) and full-text search (FTS),
but does not support native hybrid search. Hybrid search can be achieved by
combining results from both modes using RRF fusion.
"""

import contextlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import duckdb

    DUCKDB_AVAILABLE = True
    logger.info("DuckDB available - DuckDBVectorStore ready")
except ImportError as e:
    DUCKDB_AVAILABLE = False
    logger.warning(f"duckdb not installed - DuckDBVectorStore unavailable: {e}")

from ...core.chunk import Chunk
from ...core.search_result import SearchResult
from ..base import BaseVectorStore
from ..capabilities import VectorStoreCapabilities


class DuckDBVectorStore(BaseVectorStore):
    """DuckDB vector store supporting both VSS and FTS.

    DuckDB provides:
    - Vector Similarity Search (VSS) via the vss extension
    - Full-Text Search (FTS) via the fts extension
    - No native hybrid search support

    The database file serves as the persistent storage.
    """

    def __init__(
        self,
        database_path: str = ":memory:",
        table_name: str = "chunks",
        vector_dimension: int = 384,
    ):
        """Initialize DuckDB vector store.

        Args:
            database_path: Path to database file, or ":memory:" for in-memory
            table_name: Name of the table to store chunks
            vector_dimension: Dimension of embedding vectors
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is not installed. Install with: pip install duckdb")

        self.database_path = database_path
        self.table_name = table_name
        self.vector_dimension = vector_dimension
        self._connection = None

        # Initialize capabilities
        self._capabilities = VectorStoreCapabilities(
            supports_vector=True,
            supports_fulltext=True,
            supports_hybrid=False,  # DuckDB doesn't support native hybrid
        )

        # Connect and initialize
        self._ensure_connection()
        self._initialize_table()
        self._load_extensions()

        logger.info(
            f"Initialized DuckDBVectorStore: db={database_path}, "
            f"table={table_name}, dim={vector_dimension}"
        )

    def _ensure_connection(self):
        """Ensure database connection is active."""
        if self._connection is None:
            self._connection = duckdb.connect(self.database_path)

    def _initialize_table(self):
        """Create the chunks table if it doesn't exist."""
        schema = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            content VARCHAR NOT NULL,
            embedding FLOAT[{self.vector_dimension}],
            source_doc_id VARCHAR,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self._connection.execute(schema)
        logger.debug(f"Ensured table {self.table_name} exists")

    def _load_extensions(self):
        """Load required DuckDB extensions."""
        try:
            # Load VSS extension for vector similarity search
            self._connection.execute("INSTALL vss; LOAD vss;")
            logger.debug("Loaded VSS extension for vector similarity search")

            # Load FTS extension for full-text search
            self._connection.execute("INSTALL fts; LOAD fts;")
            logger.debug("Loaded FTS extension for full-text search")

        except Exception as e:
            logger.error(f"Failed to load DuckDB extensions: {e}")
            raise

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        """Get capabilities - supports vector and full-text search."""
        return self._capabilities

    def _clean_metadata(self, metadata: dict[str, Any]) -> str:
        """Convert metadata dict to JSON string for storage."""
        if metadata is None:
            return "{}"
        return json.dumps(metadata, ensure_ascii=False)

    def _parse_metadata(self, metadata_json: str) -> dict[str, Any]:
        """Parse metadata JSON string back to dict."""
        if not metadata_json or metadata_json == "{}":
            return {}
        try:
            return json.loads(metadata_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata JSON: {metadata_json}")
            return {}

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to DuckDB store.

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

        # Prepare data for insertion
        data = []
        for chunk in chunks:
            data.append(
                (
                    chunk.id,
                    chunk.content,
                    chunk.embedding,
                    chunk.source_doc_id or "",
                    self._clean_metadata(chunk.metadata),
                )
            )

        # Insert chunks
        insert_sql = f"""
        INSERT OR REPLACE INTO {self.table_name}
        (id, content, embedding, source_doc_id, metadata)
        VALUES (?, ?, ?, ?, ?)
        """

        try:
            self._connection.executemany(insert_sql, data)
            logger.info(f"Added {len(chunks)} chunks to DuckDB table {self.table_name}")

            # Create/update vector index if we have enough data
            self._ensure_vector_index()

            # Create/update full-text index
            self._ensure_fulltext_index()

        except Exception as e:
            logger.error(f"Failed to add chunks to DuckDB: {e}")
            raise

    def _ensure_vector_index(self):
        """Create vector index for efficient similarity search."""
        try:
            # Enable experimental persistence for HNSW if using persistent database
            if self.database_path != ":memory:":
                self._connection.execute("SET hnsw_enable_experimental_persistence = true;")

            # Create HNSW index for vector similarity search
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_vss_idx
            ON {self.table_name} USING HNSW (embedding)
            """
            self._connection.execute(index_sql)
            logger.debug(f"Created/ensured vector index for {self.table_name}")

        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

    def _ensure_fulltext_index(self):
        """Create full-text search index."""
        try:
            # Create FTS index on content column
            # DuckDB creates a table named fts_main_{table_name}
            index_sql = f"""
            PRAGMA create_fts_index(
                '{self.table_name}',
                'id',
                'content'
            )
            """
            self._connection.execute(index_sql)
            logger.debug(f"Created/ensured full-text index for {self.table_name}")

        except Exception as e:
            logger.warning(f"Failed to create full-text index: {e}")

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using vector similarity (VSS).

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results, sorted by similarity descending
        """
        search_sql = f"""
        SELECT id, content, embedding, source_doc_id, metadata,
               list_cosine_distance(embedding, ?::FLOAT[{self.vector_dimension}]) as distance
        FROM {self.table_name}
        ORDER BY distance ASC
        LIMIT ?
        """

        try:
            results = self._connection.execute(search_sql, [query_vector, top_k]).fetchall()

            search_results = []
            for row in results:
                chunk_id, content, embedding, source_doc_id, metadata_json, distance = row

                # Parse metadata
                metadata = self._parse_metadata(metadata_json)

                # Reconstruct chunk
                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    embedding=embedding,
                    source_doc_id=source_doc_id,
                    metadata=metadata,
                )

                # Convert distance to similarity score (cosine distance to similarity)
                # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance
                score = 1.0 - distance if distance is not None else 0.0

                search_results.append(SearchResult(chunk=chunk, score=score))

            logger.debug(f"Vector search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"DuckDB vector search failed: {e}")
            return []

    def search_fulltext(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        """Search using full-text search (FTS) with BM25.

        Args:
            query_text: Query text for keyword matching
            top_k: Number of results to return

        Returns:
            List of search results, sorted by BM25 score descending
        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided to search_fulltext")
            return []

        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}, using default 5")
            top_k = 5

        fts_table = f"fts_main_{self.table_name}"

        search_sql = f"""
        SELECT c.id, c.content, c.embedding, c.source_doc_id, c.metadata,
               {fts_table}.match_bm25(c.id, ?) as score
        FROM {self.table_name} c
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """

        try:
            results = self._connection.execute(search_sql, [query_text, top_k]).fetchall()

            if not results:
                logger.debug("No full-text search results found")
                return []

            # Single-pass processing with improved normalization
            search_results = []
            raw_scores = [float(row[5] or 0.0) for row in results]

            # Use max score for normalization (preserves relative differences)
            max_score = max(raw_scores) if raw_scores else 1.0

            # Avoid division by zero
            if max_score <= 0:
                max_score = 1.0

            for row in results:
                chunk_id, content, embedding, source_doc_id, metadata_json, raw_score = row

                # Parse metadata
                metadata = self._parse_metadata(metadata_json)

                # Reconstruct chunk
                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    embedding=embedding,
                    source_doc_id=source_doc_id,
                    metadata=metadata,
                )

                # Normalize score: divide by max to get [0, 1] range while preserving relative order
                normalized_score = float(raw_score or 0.0) / max_score
                normalized_score = max(0.0, min(1.0, normalized_score))

                search_results.append(SearchResult(chunk=chunk, score=normalized_score))

            logger.debug(f"Full-text search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"DuckDB full-text search failed: {e}")
            return []

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        if not chunk_ids:
            return

        # Prepare placeholders for SQL
        placeholders = ",".join("?" for _ in chunk_ids)

        delete_sql = f"""
        DELETE FROM {self.table_name}
        WHERE id IN ({placeholders})
        """

        try:
            self._connection.execute(delete_sql, chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from DuckDB table {self.table_name}")

        except Exception as e:
            logger.error(f"Failed to delete chunks from DuckDB: {e}")
            raise

    def count(self) -> int:
        """Get the total number of chunks in this store.

        Returns:
            Number of chunks currently stored
        """
        try:
            result = self._connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to count chunks in DuckDB: {e}")
            return 0

    def persist(self, path: str) -> None:
        """Persist database to file.

        Args:
            path: File path to save database to
        """
        if self.database_path == ":memory:":
            # For in-memory databases, we need to export to file
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Export to a new database file
            export_sql = f"EXPORT DATABASE '{path}'"
            try:
                self._connection.execute(export_sql)
                logger.info(f"Exported DuckDB database to {path}")
            except Exception as e:
                logger.error(f"Failed to export DuckDB database: {e}")
                raise
        else:
            # Already persistent, just ensure data is written
            self._connection.execute("CHECKPOINT")
            logger.info("DuckDB database checkpoint completed")

    def load(self, path: str) -> None:
        """Load database from file.

        Args:
            path: File path to load database from

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"DuckDB database file not found: {path}")

        # Close current connection
        if self._connection:
            self._connection.close()

        # Connect to the new database and detect vector dimension
        self.database_path = path
        temp_conn = duckdb.connect(path)
        try:
            # Try to detect vector dimension from existing data
            result = temp_conn.execute(f"""
                SELECT length(embedding) as dim
                FROM {self.table_name}
                LIMIT 1
            """).fetchone()
            if result:
                self.vector_dimension = result[0]
        except Exception:
            # If detection fails, keep current dimension
            pass
        temp_conn.close()

        # Connect with proper configuration
        self._connection = duckdb.connect(path)
        self._load_extensions()

        logger.info(f"Loaded DuckDB database from {path} (detected dim={self.vector_dimension})")

    def __del__(self):
        """Cleanup connection on destruction."""
        if self._connection:
            with contextlib.suppress(Exception):
                self._connection.close()
