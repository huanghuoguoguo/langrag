"""
SQLite-based Key-Value Store.

This module provides a persistent key-value store using SQLite,
suitable for storing document metadata, parent-child relationships,
and other auxiliary data in RAG applications.

Features:
- Thread-safe operations with locking
- Connection reuse for better performance
- Context manager support for proper resource cleanup
- Configurable connection timeout

Example:
    Using context manager (recommended):

    >>> with SQLiteKV(db_path="./kv.db") as kv:
    ...     kv.mset({"key1": "value1", "key2": "value2"})
    ...     values = kv.mget(["key1", "key2"])
    >>> # Connection automatically closed

    Manual resource management:

    >>> kv = SQLiteKV(db_path="./kv.db")
    >>> try:
    ...     kv.mset({"key1": "value1"})
    ... finally:
    ...     kv.close()
"""

import contextlib
import logging
import sqlite3
import threading
from typing import Any

from .base import BaseKVStore

logger = logging.getLogger(__name__)


class SQLiteKV(BaseKVStore):
    """
    Persistent Key-Value Store using SQLite.

    This implementation uses a single reusable connection with proper
    thread synchronization to avoid "database is locked" errors in
    concurrent scenarios.

    Attributes:
        db_path: Path to the SQLite database file
        table_name: Name of the table storing key-value pairs
        timeout: Connection timeout in seconds (default: 30.0)

    Example:
        >>> with SQLiteKV("data.db", timeout=60.0) as store:
        ...     store.mset({"user:1": '{"name": "Alice"}'})
        ...     data = store.mget(["user:1"])
    """

    def __init__(
        self,
        db_path: str = "kv_store.db",
        table_name: str = "kv_store",
        timeout: float = 30.0
    ):
        """
        Initialize SQLite key-value store.

        Args:
            db_path: Path to SQLite database file
            table_name: Table name for key-value storage
            timeout: Connection timeout in seconds (default: 30.0)
                     Helps prevent deadlocks in concurrent scenarios.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.timeout = timeout
        self._lock = threading.RLock()  # Use RLock for reentrant locking
        self._closed = False

        # Create a single reusable connection
        self._connection = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False  # Allow multi-thread access with our lock
        )
        # Enable WAL mode for better concurrent read performance
        self._connection.execute("PRAGMA journal_mode=WAL")

        self._init_db()
        logger.debug(f"SQLiteKV initialized: {db_path}, table={table_name}")

    def _init_db(self) -> None:
        """Initialize the database table if it doesn't exist."""
        with self._lock:
            self._connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            self._connection.commit()

    def _check_closed(self) -> None:
        """Raise an error if the connection has been closed."""
        if self._closed:
            raise RuntimeError(
                "Cannot perform operation: SQLiteKV connection has been closed. "
                "Create a new SQLiteKV instance to continue."
            )

    def mget(self, keys: list[str]) -> list[Any | None]:
        """
        Get multiple values by keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            List of values in the same order as keys.
            None for keys that don't exist.
        """
        self._check_closed()
        if not keys:
            return []

        placeholders = ",".join("?" * len(keys))
        query = f"SELECT key, value FROM {self.table_name} WHERE key IN ({placeholders})"

        with self._lock:
            cursor = self._connection.execute(query, keys)
            results_map = dict(cursor.fetchall())

        return [results_map.get(k) for k in keys]

    def mset(self, data: dict[str, Any]) -> None:
        """
        Set multiple key-value pairs.

        Args:
            data: Dictionary of key-value pairs to store.
                  Values are converted to strings.
        """
        self._check_closed()
        if not data:
            return

        query = f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)"
        params = [(k, str(v)) for k, v in data.items()]

        with self._lock:
            self._connection.executemany(query, params)
            self._connection.commit()

    def delete(self, keys: list[str]) -> None:
        """
        Delete multiple keys.

        Args:
            keys: List of keys to delete
        """
        self._check_closed()
        if not keys:
            return

        placeholders = ",".join("?" * len(keys))
        query = f"DELETE FROM {self.table_name} WHERE key IN ({placeholders})"

        with self._lock:
            self._connection.execute(query, keys)
            self._connection.commit()

    def close(self) -> None:
        """
        Close the database connection.

        This method should be called when done using the store to ensure
        proper resource cleanup. Alternatively, use the context manager pattern.

        Note:
            After calling close(), the store instance cannot be used.
            Any subsequent operations will raise RuntimeError.
        """
        if self._closed:
            return

        with self._lock:
            try:
                self._connection.close()
                logger.debug(f"SQLiteKV connection closed: {self.db_path}")
            except Exception as e:
                logger.warning(f"Error closing SQLiteKV connection: {e}")
            finally:
                self._closed = True

    def __enter__(self) -> "SQLiteKV":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __del__(self):
        """Clean up database connection on garbage collection."""
        # Note: __del__ is not guaranteed to be called
        if not self._closed:
            with contextlib.suppress(Exception):
                self.close()
