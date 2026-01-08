
import pytest
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from langrag.datasource.kv.sqlite import SQLiteKV

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
            futures = [executor.submit(write_task, i) for i in range(count)]
            
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
