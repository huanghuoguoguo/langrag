
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from langrag.datasource.kv.sqlite import (
    SQLiteKV,
)


class TestSQLiteKV:

    @pytest.fixture
    def db_path(self):
        # Create a temp file path
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_basic_ops(self, db_path):
        kv = SQLiteKV(db_path=db_path)

        # Test Set / Get
        kv.mset({"k1": "v1", "k2": "v2"})
        results = kv.mget(["k1", "missing", "k2"])
        assert results == ["v1", None, "v2"]

        # Test Delete
        kv.delete(["k1"])
        results = kv.mget(["k1", "k2"])
        assert results == [None, "v2"]

    def test_persistence(self, db_path):
        # Instance 1: Write
        kv1 = SQLiteKV(db_path=db_path)
        kv1.mset({"persistent_key": "persistent_val"})
        del kv1  # "Close" it

        # Instance 2: Read
        kv2 = SQLiteKV(db_path=db_path)
        val = kv2.mget(["persistent_key"])[0]
        assert val == "persistent_val"

    def test_concurrency(self, db_path):
        # Stress test with multiple threads writing to the same DB
        kv = SQLiteKV(db_path=db_path)

        def write_task(i):
            key = f"key_{i}"
            val = f"val_{i}"
            kv.mset({key: val})
            return key, val

        count = 100
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(write_task, range(count)))

        # Verify all writes succeeded
        keys = [f"key_{i}" for i in range(count)]
        values = kv.mget(keys)

        for i, val in enumerate(values):
            assert val == f"val_{i}", f"Concurrency failure at index {i}"

    def test_special_chars(self, db_path):
        kv = SQLiteKV(db_path=db_path)

        special_data = {
            "emoji": "🚀🎉",
            "sql_injection_attempt": "'; DROP TABLE kv_store; --",
            "newline": "line1\nline2",
            "empty": ""
        }

        kv.mset(special_data)

        for k, expected_v in special_data.items():
            actual_v = kv.mget([k])[0]
            assert actual_v == expected_v, f"Failed for key: {k}"

    def test_large_value(self, db_path):
        kv = SQLiteKV(db_path=db_path)
        large_val = "x" * 100000 # 100KB
        kv.mset({"large": large_val})
        assert kv.mget(["large"])[0] == large_val
        kv.close()

    def test_context_manager(self, db_path):
        """Test that context manager properly opens and closes connection."""
        with SQLiteKV(db_path=db_path) as kv:
            assert kv._closed is False
            kv.mset({"ctx_key": "ctx_value"})
            assert kv.mget(["ctx_key"])[0] == "ctx_value"

        # After exiting context, connection should be closed
        assert kv._closed is True

    def test_close_method(self, db_path):
        """Test that close() method properly closes connection."""
        kv = SQLiteKV(db_path=db_path)
        assert kv._closed is False

        kv.close()

        assert kv._closed is True

    def test_close_idempotent(self, db_path):
        """Test that calling close() multiple times is safe."""
        kv = SQLiteKV(db_path=db_path)

        kv.close()
        kv.close()  # Should not raise
        kv.close()  # Should not raise

        assert kv._closed is True

    def test_operations_after_close_raise_error(self, db_path):
        """Test that operations after close() raise RuntimeError."""
        kv = SQLiteKV(db_path=db_path)
        kv.close()

        with pytest.raises(RuntimeError, match="connection has been closed"):
            kv.mget(["key"])

        with pytest.raises(RuntimeError, match="connection has been closed"):
            kv.mset({"key": "value"})

        with pytest.raises(RuntimeError, match="connection has been closed"):
            kv.delete(["key"])

    def test_context_manager_with_exception(self, db_path):
        """Test that context manager closes connection even when exception occurs."""
        kv = None
        try:
            with SQLiteKV(db_path=db_path) as kv:
                kv.mset({"before_error": "value"})
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Connection should still be closed after exception
        assert kv is not None
        assert kv._closed is True

    def test_timeout_parameter(self, db_path):
        """Test that timeout parameter is accepted."""
        kv = SQLiteKV(db_path=db_path, timeout=60.0)
        assert kv.timeout == 60.0
        kv.mset({"key": "value"})
        assert kv.mget(["key"])[0] == "value"
        kv.close()

    def test_connection_reuse(self, db_path):
        """Test that connection is reused across operations."""
        kv = SQLiteKV(db_path=db_path)

        # Get connection id before operations
        conn_id_before = id(kv._connection)

        # Perform multiple operations
        kv.mset({"k1": "v1"})
        kv.mget(["k1"])
        kv.mset({"k2": "v2"})
        kv.delete(["k1"])

        # Connection should be the same object
        conn_id_after = id(kv._connection)
        assert conn_id_before == conn_id_after

        kv.close()

    def test_persistence_with_context_manager(self, db_path):
        """Test data persists after context manager exits."""
        # Write with context manager
        with SQLiteKV(db_path=db_path) as kv:
            kv.mset({"persist_key": "persist_value"})

        # Read with new instance
        with SQLiteKV(db_path=db_path) as kv:
            value = kv.mget(["persist_key"])[0]
            assert value == "persist_value"


class TestSQLiteKVTableNameValidation:
    """Tests for SQLiteKV table name validation."""

    @pytest.fixture
    def db_path(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_valid_table_name(self, db_path):
        """Test that valid table names are accepted."""
        kv = SQLiteKV(db_path=db_path, table_name="valid_table")
        assert kv.table_name == "valid_table"
        kv.close()

    def test_invalid_table_name_raises(self, db_path):
        """Test that invalid table names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid table name"):
            SQLiteKV(db_path=db_path, table_name="invalid-name")

    def test_sql_injection_attempt_blocked(self, db_path):
        """Test that SQL injection attempts are blocked."""
        with pytest.raises(ValueError, match="Invalid table name"):
            SQLiteKV(db_path=db_path, table_name="'; DROP TABLE kv_store;--")

    def test_reserved_word_raises(self, db_path):
        """Test that SQL reserved words are rejected."""
        with pytest.raises(ValueError, match="reserved word"):
            SQLiteKV(db_path=db_path, table_name="select")

    def test_empty_table_name_raises(self, db_path):
        """Test that empty table name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SQLiteKV(db_path=db_path, table_name="")

    def test_default_table_name_works(self, db_path):
        """Test that default table name works."""
        kv = SQLiteKV(db_path=db_path)  # Uses default "kv_store"
        assert kv.table_name == "kv_store"
        kv.mset({"key": "value"})
        assert kv.mget(["key"])[0] == "value"
        kv.close()

