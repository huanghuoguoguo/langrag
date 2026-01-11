"""
DuckDB Vector Store with Full-Text Search and Vector Search Support.

This module provides a DuckDB-based vector store implementation that supports:
- Vector similarity search using HNSW indexing
- Full-Text Search (FTS) using BM25 ranking
- Hybrid search combining both modalities with RRF fusion

Architecture:
------------
DuckDB extensions used:
- VSS (Vector Similarity Search): For HNSW-based vector search
- FTS (Full-Text Search): For BM25-based keyword search
- JSON: For metadata storage

FTS Index Naming Convention:
----------------------------
DuckDB FTS creates an index with a specific naming pattern:
- Index name: fts_main_{table_name}
- BM25 function: fts_main_{table_name}.match_bm25(id_column, query)

Example:
    For table 'documents', the BM25 function is:
    fts_main_documents.match_bm25(id, 'search query')
"""

import contextlib
import json
import logging
import re

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.utils.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

try:
    import duckdb
    from duckdb import CatalogException
    from duckdb import Error as DuckDBError
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    # Provide fallback types for type hints when duckdb is not installed
    CatalogException = Exception
    DuckDBError = Exception


# Regex pattern for valid SQL identifiers (table names, column names)
# Allows: letters, numbers, underscores; must start with letter or underscore
SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Maximum length for table names (conservative limit)
MAX_TABLE_NAME_LENGTH = 128


class VectorDimensionError(ValueError):
    """Raised when vector dimensions are inconsistent or invalid."""
    pass


def validate_table_name(table_name: str) -> None:
    """
    Validate a table name to prevent SQL injection.

    Args:
        table_name: The table name to validate

    Raises:
        ValueError: If the table name is invalid or potentially unsafe
    """
    if not table_name:
        raise ValueError("Table name cannot be empty")

    if len(table_name) > MAX_TABLE_NAME_LENGTH:
        raise ValueError(
            f"Table name too long: {len(table_name)} characters "
            f"(max: {MAX_TABLE_NAME_LENGTH})"
        )

    if not SQL_IDENTIFIER_PATTERN.match(table_name):
        raise ValueError(
            f"Invalid table name: '{table_name}'. "
            "Table names must start with a letter or underscore, "
            "and contain only letters, numbers, and underscores."
        )

    # Check for SQL reserved words (basic set)
    reserved_words = {
        "select", "insert", "update", "delete", "drop", "create", "alter",
        "table", "index", "from", "where", "and", "or", "not", "null",
        "primary", "key", "foreign", "references", "constraint", "unique",
        "default", "check", "in", "between", "like", "order", "by", "group",
        "having", "limit", "offset", "join", "inner", "outer", "left", "right",
        "on", "as", "distinct", "all", "union", "except", "intersect",
    }

    if table_name.lower() in reserved_words:
        raise ValueError(
            f"Invalid table name: '{table_name}' is a SQL reserved word. "
            "Please choose a different name."
        )


class DuckDBVector(BaseVector):
    """
    DuckDB Vector Store with Vector and Full-Text Search Support.

    Features:
    - Vector similarity search with HNSW indexing (cosine distance)
    - Full-Text Search (FTS) with BM25 ranking
    - Hybrid search combining vector + FTS with Reciprocal Rank Fusion
    - Context manager support for proper resource cleanup

    Note:
        FTS index is created automatically after first data insertion.
        The index uses Porter stemmer, English stopwords, and lowercase normalization.

    Example:
        Using context manager (recommended):

        >>> with DuckDBVector(dataset, database_path="./vectors.db") as store:
        ...     store.add_texts(documents)
        ...     results = store.search("query", query_vector, search_type="hybrid")
        >>> # Connection automatically closed

        Manual resource management:

        >>> store = DuckDBVector(dataset, database_path="./vectors.db")
        >>> try:
        ...     store.add_texts(documents)
        ...     results = store.search("query", query_vector)
        ... finally:
        ...     store.close()
    """

    def __init__(
        self,
        dataset: Dataset,
        database_path: str = "./duckdb_vector.db",
        table_name: str | None = None
    ):
        """
        Initialize DuckDB vector store.

        Args:
            dataset: Dataset configuration object
            database_path: Path to DuckDB database file
            table_name: Table name for storage (defaults to dataset.collection_name)

        Raises:
            ValueError: If table_name contains invalid characters or is a reserved word
        """
        super().__init__(dataset)
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.database_path = database_path
        self.table_name = table_name or self.dataset.collection_name

        # Validate table name to prevent SQL injection
        validate_table_name(self.table_name)

        self._connection = duckdb.connect(self.database_path)
        self._fts_index_created = False
        self._closed = False
        self._vector_dim: int | None = None  # Track expected vector dimension

        # Load required extensions
        self._load_extensions()

    def _load_extensions(self) -> None:
        """Load DuckDB extensions for vector search, FTS, and JSON support."""
        try:
            self._connection.execute("INSTALL vss; LOAD vss;")
            self._connection.execute("INSTALL fts; LOAD fts;")
            self._connection.execute("INSTALL json; LOAD json;")
            logger.debug("DuckDB extensions loaded: VSS, FTS, JSON")
        except Exception as e:
            logger.warning(f"Failed to load DuckDB extensions: {e}")

    def _ensure_table_exists(self, dim: int) -> None:
        """
        Ensure the table exists with the correct schema.

        Args:
            dim: Vector dimension for the embedding column
        """
        schema = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            content VARCHAR NOT NULL,
            embedding FLOAT[{dim}],
            metadata JSON,
            doc_id VARCHAR
        )
        """
        self._connection.execute(schema)

    def _create_vector_index(self) -> None:
        """Create HNSW index for vector similarity search."""
        try:
            self._connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_vec_{self.table_name} "
                f"ON {self.table_name} USING HNSW (embedding)"
            )
            logger.debug(f"HNSW index created for {self.table_name}")
        except Exception as e:
            logger.warning(f"HNSW index creation warning: {e}")

    def _create_fts_index(self) -> None:
        """
        Create Full-Text Search index on the content column.

        The FTS index uses:
        - Porter stemmer for word normalization
        - English stopwords removal
        - Case-insensitive matching
        - Accent stripping
        """
        if self._fts_index_created:
            return

        try:
            # Create FTS index with stemming and stopwords
            self._connection.execute(f"""
                PRAGMA create_fts_index(
                    '{self.table_name}',
                    'id',
                    'content',
                    stemmer='porter',
                    stopwords='english',
                    strip_accents=1,
                    lower=1
                )
            """)
            self._fts_index_created = True
            logger.info(f"FTS index created for {self.table_name}")
        except DuckDBError as e:
            error_msg = str(e)
            # If index already exists, mark as created (not an error)
            if "already exists" in error_msg.lower():
                self._fts_index_created = True
                logger.debug(f"FTS index already exists for {self.table_name}")
            else:
                logger.warning(f"FTS index creation failed: {e}")

    def _validate_vector_dimensions(self, texts: list[Document]) -> int:
        """
        Validate that all documents have vectors with consistent dimensions.

        Args:
            texts: List of documents to validate

        Returns:
            The vector dimension (all documents must match)

        Raises:
            VectorDimensionError: If vectors have inconsistent dimensions or are missing
        """
        if not texts:
            raise VectorDimensionError("Cannot validate empty document list")

        # Get first document's dimension as reference
        first_doc = texts[0]
        if first_doc.vector is None:
            raise VectorDimensionError(
                f"Document '{first_doc.id}' has no vector. "
                "All documents must have vectors for vector store insertion."
            )

        expected_dim = len(first_doc.vector)
        if expected_dim == 0:
            raise VectorDimensionError(
                f"Document '{first_doc.id}' has empty vector (dimension 0)."
            )

        # Validate all documents have same dimension
        for i, doc in enumerate(texts[1:], start=1):
            if doc.vector is None:
                raise VectorDimensionError(
                    f"Document '{doc.id}' (index {i}) has no vector. "
                    "All documents must have vectors for vector store insertion."
                )

            doc_dim = len(doc.vector)
            if doc_dim == 0:
                raise VectorDimensionError(
                    f"Document '{doc.id}' (index {i}) has empty vector (dimension 0)."
                )
            if doc_dim != expected_dim:
                raise VectorDimensionError(
                    f"Vector dimension mismatch: document '{doc.id}' (index {i}) "
                    f"has dimension {doc_dim}, expected {expected_dim}."
                )

        # If we have a stored dimension, validate against it
        if self._vector_dim is not None and expected_dim != self._vector_dim:
            raise VectorDimensionError(
                f"Vector dimension mismatch: new documents have dimension {expected_dim}, "
                f"but the collection was created with dimension {self._vector_dim}."
            )

        return expected_dim

    def create(self, texts: list[Document], **kwargs) -> None:
        """
        Create or replace collection with initial documents.

        Args:
            texts: List of documents to insert
            **kwargs: Additional arguments (unused)

        Raises:
            VectorDimensionError: If documents have inconsistent or missing vectors
        """
        self._check_closed()
        if not texts:
            return

        # Validate vector dimensions and get the dimension
        dim = self._validate_vector_dimensions(texts)
        self._vector_dim = dim

        # Create table schema
        self._ensure_table_exists(dim)

        # Create vector index
        self._create_vector_index()

        # Insert documents
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **_kwargs) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of documents to add
            **kwargs: Additional arguments (unused)

        Raises:
            VectorDimensionError: If documents have inconsistent or incompatible vectors
        """
        self._check_closed()
        if not texts:
            return

        # Ensure table exists
        try:
            self._connection.execute(f"SELECT 1 FROM {self.table_name} LIMIT 0")
        except CatalogException:
            # Table doesn't exist, create it
            logger.debug(f"Table '{self.table_name}' not found, creating...")
            self.create(texts)
            return

        # Validate vector dimensions
        dim = self._validate_vector_dimensions(texts)

        # Set _vector_dim if not already set (e.g., reconnecting to existing table)
        if self._vector_dim is None:
            self._vector_dim = dim

        # Prepare and insert data
        data = []
        for doc in texts:
            meta_json = json.dumps(doc.metadata)
            data.append((doc.id, doc.page_content, doc.vector, meta_json, doc.id))

        sql = f"INSERT OR REPLACE INTO {self.table_name} VALUES (?, ?, ?, ?, ?)"
        self._connection.executemany(sql, data)

        # Create FTS index after data insertion (FTS needs data to index)
        self._create_fts_index()

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """
        Search the vector store.

        Args:
            query: Text query for FTS search
            query_vector: Vector for similarity search
            top_k: Number of results to return
            **kwargs: Additional arguments
                - search_type: 'similarity', 'keyword', or 'hybrid'

        Returns:
            List of matching documents with scores in metadata
        """
        self._check_closed()
        search_type = kwargs.get('search_type', 'similarity')

        if search_type == 'hybrid' and query_vector:
            return self._search_hybrid(query, query_vector, top_k)
        elif search_type == 'keyword':
            return self._search_keyword(query, top_k)
        else:
            return self._search_vector(query_vector, top_k)

    def _search_vector(self, query_vector: list[float] | None, top_k: int) -> list[Document]:
        """
        Perform vector similarity search using cosine distance.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of documents ordered by similarity (highest first)
        """
        if not query_vector:
            return []

        sql = f"""
        SELECT id, content, metadata,
               list_cosine_distance(embedding, ?::FLOAT[{len(query_vector)}]) as dist
        FROM {self.table_name}
        ORDER BY dist ASC
        LIMIT ?
        """
        rows = self._connection.execute(sql, [query_vector, top_k]).fetchall()
        return self._rows_to_docs(rows, score_strategy='distance')

    def _search_keyword(self, query: str, top_k: int) -> list[Document]:
        """
        Perform Full-Text Search using BM25 ranking.

        Args:
            query: Text query for keyword search
            top_k: Number of results to return

        Returns:
            List of documents ordered by BM25 score (highest first)
        """
        # Ensure FTS index exists
        self._create_fts_index()

        fts_table = f"fts_main_{self.table_name}"

        sql = f"""
        SELECT id, content, metadata,
               {fts_table}.match_bm25(id, ?) as score
        FROM {self.table_name}
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """

        try:
            rows = self._connection.execute(sql, [query, top_k]).fetchall()
            return self._rows_to_docs(rows, score_strategy='score')
        except Exception as e:
            logger.error(f"FTS query failed: {e}")
            return []

    def _search_hybrid(
        self,
        query: str,
        query_vector: list[float],
        top_k: int
    ) -> list[Document]:
        """
        Perform hybrid search combining vector and FTS with RRF fusion.

        The hybrid search:
        1. Retrieves top 2*top_k results from vector search
        2. Retrieves top 2*top_k results from FTS search
        3. Fuses results using Reciprocal Rank Fusion (RRF)
        4. Returns top top_k fused results

        Args:
            query: Text query for FTS
            query_vector: Query embedding for vector search
            top_k: Number of final results

        Returns:
            List of documents with fused scores
        """
        # Expand retrieval for fusion
        expanded_k = top_k * 2

        # Get vector search results
        vector_results = self._search_vector(query_vector, expanded_k)

        # Get FTS results
        fts_results = self._search_keyword(query, expanded_k)

        # If one source has no results, return the other
        if not fts_results:
            return vector_results[:top_k]
        if not vector_results:
            return fts_results[:top_k]

        # Convert to SearchResult objects for RRF
        vector_search_results = [
            SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
            for doc in vector_results
        ]
        fts_search_results = [
            SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
            for doc in fts_results
        ]

        # Fuse results using RRF
        fused_ranking = reciprocal_rank_fusion(
            [vector_search_results, fts_search_results],
            top_k=top_k
        )

        # Convert back to Document list with fused scores
        results = []
        for search_result in fused_ranking:
            doc = search_result.chunk
            doc.metadata['score'] = search_result.score
            doc.metadata['search_type'] = 'hybrid'
            results.append(doc)

        return results

    def _rows_to_docs(
        self,
        rows: list[tuple],
        score_strategy: str = 'score'
    ) -> list[Document]:
        """
        Convert database rows to Document objects.

        Args:
            rows: List of (id, content, metadata, score/distance) tuples
            score_strategy: How to interpret the fourth column:
                - 'distance': Convert cosine distance to similarity
                - 'score': Use raw BM25 score

        Returns:
            List of Document objects with scores in metadata
        """
        docs = []
        for row in rows:
            doc_id, content, meta_json, val = row

            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            if score_strategy == 'distance':
                # Convert cosine distance to similarity score
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                # Convert to 0-1 similarity where 1 is most similar
                score = 1.0 / (1.0 + val) if val is not None else 0.0
            else:
                # Raw BM25 score (higher is better)
                score = val if val is not None else 0.0

            meta['score'] = score

            docs.append(Document(
                id=doc_id,
                page_content=content,
                metadata=meta
            ))

        return docs

    def delete_by_ids(self, ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        self._check_closed()
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._connection.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            ids
        )

    def delete(self) -> None:
        """Delete the entire collection (drop table)."""
        self._check_closed()
        try:
            # Drop FTS index first
            with contextlib.suppress(Exception):
                self._connection.execute(f"PRAGMA drop_fts_index('{self.table_name}')")

            # Drop the main table
            self._connection.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self._fts_index_created = False
        except Exception as e:
            logger.warning(f"Error deleting table: {e}")

    def close(self) -> None:
        """
        Explicitly close the database connection.

        This method should be called when done using the vector store to ensure
        proper resource cleanup. Alternatively, use the context manager pattern:

            with DuckDBVector(dataset, path) as store:
                store.add_texts(docs)
                results = store.search(...)
            # Connection automatically closed here

        Note:
            After calling close(), the vector store instance cannot be used.
            Any subsequent operations will raise RuntimeError.
        """
        if self._closed:
            return

        try:
            self._connection.close()
            logger.debug(f"DuckDB connection closed for {self.table_name}")
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")
        finally:
            self._closed = True

    def _check_closed(self) -> None:
        """Raise an error if the connection has been closed."""
        if self._closed:
            raise RuntimeError(
                "Cannot perform operation: DuckDB connection has been closed. "
                "Create a new DuckDBVector instance to continue."
            )

    def __enter__(self) -> "DuckDBVector":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __del__(self):
        """Clean up database connection on garbage collection."""
        # Note: __del__ is not guaranteed to be called, so users should
        # prefer using close() explicitly or the context manager pattern.
        with contextlib.suppress(Exception):
            if not self._closed:
                self.close()
