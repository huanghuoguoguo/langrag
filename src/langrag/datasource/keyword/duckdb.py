import contextlib
import json
import logging
import re

from langrag.datasource.keyword.base import BaseKeyword
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

logger = logging.getLogger(__name__)

try:
    import duckdb
    from duckdb import CatalogException
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    CatalogException = Exception  # Fallback for type hints


# Regex pattern for valid SQL identifiers
SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
MAX_TABLE_NAME_LENGTH = 128


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

    # Check for SQL reserved words
    reserved_words = {
        "select", "insert", "update", "delete", "drop", "create", "alter",
        "table", "index", "from", "where", "and", "or", "not", "null",
    }

    if table_name.lower() in reserved_words:
        raise ValueError(
            f"Invalid table name: '{table_name}' is a SQL reserved word."
        )


class DuckDBKeyword(BaseKeyword):
    """
    DuckDB implementation for Keyword/Full-Text Search.
    Uses DuckDB's FTS extension via BM25.

    Example:
        Using context manager (recommended):

        >>> with DuckDBKeyword(dataset, database_path="./fts.db") as store:
        ...     store.add_texts(documents)
        ...     results = store.search("query")
        >>> # Connection automatically closed
    """

    def __init__(
        self,
        dataset: Dataset,
        database_path: str = ":memory:",
        table_name: str | None = None
    ):
        super().__init__(dataset)
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.database_path = database_path
        self.table_name = table_name or self.dataset.collection_name

        # Validate table name to prevent SQL injection
        validate_table_name(self.table_name)

        self._connection = duckdb.connect(self.database_path)
        self._closed = False
        self._init_db()

    def _init_db(self):
        """Initialize database table and FTS index."""
        schema = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            content VARCHAR NOT NULL,
            metadata JSON,
            doc_id VARCHAR
        )
        """
        self._connection.execute(schema)

        # Install FTS
        try:
            self._connection.execute("INSTALL fts; LOAD fts;")
            self._connection.execute(
                f"PRAGMA create_fts_index('{self.table_name}', 'id', 'content');"
            )
        except CatalogException:
            # FTS index already exists
            logger.debug(f"FTS index already exists for {self.table_name}")
        except Exception as e:
            logger.warning(f"DuckDB FTS init warning: {e}")

    def _check_closed(self) -> None:
        """Raise an error if the connection has been closed."""
        if self._closed:
            raise RuntimeError(
                "Cannot perform operation: DuckDB connection has been closed. "
                "Create a new DuckDBKeyword instance to continue."
            )

    def create(self, texts: list[Document], **kwargs) -> None:
        self._check_closed()
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **_kwargs) -> None:
        self._check_closed()
        if not texts:
            return

        data = []
        for doc in texts:
            meta_json = json.dumps(doc.metadata)
            data.append((doc.id, doc.page_content, meta_json, doc.id))

        sql = f"INSERT OR REPLACE INTO {self.table_name} VALUES (?, ?, ?, ?)"
        self._connection.executemany(sql, data)

    def search(self, query: str, top_k: int = 4, **_kwargs) -> list[Document]:
        self._check_closed()
        if not query:
            return []

        fts_table = f"fts_main_{self.table_name}"
        sql = f"""
        SELECT t.id, t.content, t.metadata, {fts_table}.match_bm25(t.id, ?) as score
        FROM {self.table_name} t
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """

        try:
            rows = self._connection.execute(sql, [query, top_k]).fetchall()
        except Exception as e:
            logger.error(f"DuckDB search failed: {e}")
            return []

        docs = []
        for row in rows:
            doc_id, content, meta_json, score = row
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            meta['score'] = score

            docs.append(Document(
                id=doc_id,
                page_content=content,
                metadata=meta
            ))

        return docs

    def delete_by_ids(self, ids: list[str]) -> None:
        self._check_closed()
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        self._connection.execute(sql, ids)

    def close(self) -> None:
        """
        Close the database connection.

        This method should be called when done using the store to ensure
        proper resource cleanup. Alternatively, use the context manager pattern.
        """
        if self._closed:
            return

        try:
            self._connection.close()
            logger.debug(f"DuckDBKeyword connection closed for {self.table_name}")
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")
        finally:
            self._closed = True

    def __enter__(self) -> "DuckDBKeyword":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __del__(self):
        """Clean up database connection on garbage collection."""
        with contextlib.suppress(Exception):
            if hasattr(self, '_closed') and not self._closed:
                self.close()
