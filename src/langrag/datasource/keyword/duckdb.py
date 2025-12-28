import contextlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.keyword.base import BaseKeyword

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

class DuckDBKeyword(BaseKeyword):
    """
    DuckDB implementation for Keyword/Full-Text Search.
    Uses DuckDB's FTS extension via BM25.
    """

    def __init__(
        self,
        dataset: Dataset,
        database_path: str = ":memory:", # or path to db file
        table_name: str | None = None
    ):
        super().__init__(dataset)
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.database_path = database_path
        # Use dataset's collection name as table name if not provided
        self.table_name = table_name or self.dataset.collection_name
        
        self._connection = duckdb.connect(self.database_path)
        self._init_db()

    def _init_db(self):
        # Create table
        # We store minimal fields required for keyword search and retrieval
        # Note: We don't store vector here.
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
             
             # Create FTS Index if not exists
             # DuckDB FTS index creation is idempotent via PRAGMA?
             # Docs differ, but typically we create it after specific table creation.
             # PRAGMA create_fts_index('table_name', 'id_col', 'content_col');
             self._connection.execute(f"PRAGMA create_fts_index('{self.table_name}', 'id', 'content');")
        except Exception as e:
             logger.warning(f"DuckDB FTS init warning: {e}")

    def create(self, texts: list[Document], **kwargs) -> None:
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        if not texts: return
        
        data = []
        for doc in texts:
            meta_json = json.dumps(doc.metadata)
            data.append((doc.id, doc.page_content, meta_json, doc.id))
            
        sql = f"INSERT OR REPLACE INTO {self.table_name} VALUES (?, ?, ?, ?)"
        self._connection.executemany(sql, data)

    def search(self, query: str, top_k: int = 4, **kwargs) -> list[Document]:
        if not query: return []
        
        # DuckDB FTS search
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
                meta = json.loads(meta_json)
            except:
                meta = {}
            
            # Normalize score if needed, BM25 is unbounded.
            # For now we just return raw score or normalized via max in batch (omitted for brevity)
            meta['score'] = score
            
            docs.append(Document(
                id=doc_id,
                page_content=content,
                metadata=meta
            ))
            
        return docs

    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids: return
        placeholders = ",".join("?" for _ in ids)
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        self._connection.execute(sql, ids)

    def __del__(self):
        if hasattr(self, '_connection'):
            self._connection.close()
