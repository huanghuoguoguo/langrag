
import sqlite3
import threading
from typing import Any, List, Optional
from pathlib import Path
from .base import BaseKVStore

class SQLiteKV(BaseKVStore):
    """
    Persistent Key-Value Store using SQLite.
    Storing (key, value) pairs in a simple table.
    """

    def __init__(self, db_path: str = "kv_store.db", table_name: str = "kv_store"):
        self.db_path = db_path
        self.table_name = table_name
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                conn.commit()

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        if not keys:
            return []
            
        placeholders = ",".join("?" * len(keys))
        query = f"SELECT key, value FROM {self.table_name} WHERE key IN ({placeholders})"
        
        results_map = {}
        with self._get_conn() as conn:
            cursor = conn.execute(query, keys)
            for k, v in cursor.fetchall():
                results_map[k] = v
                
        return [results_map.get(k) for k in keys]

    def mset(self, data: dict[str, Any]) -> None:
        if not data:
            return
            
        query = f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)"
        params = [(k, str(v)) for k, v in data.items()]
        
        with self._lock:
            with self._get_conn() as conn:
                conn.executemany(query, params)
                conn.commit()

    def delete(self, keys: List[str]) -> None:
        if not keys:
            return
            
        placeholders = ",".join("?" * len(keys))
        query = f"DELETE FROM {self.table_name} WHERE key IN ({placeholders})"
        
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(query, keys)
                conn.commit()
