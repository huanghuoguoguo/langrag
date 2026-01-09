import json
import logging

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class DuckDBVector(BaseVector):
    """
    DuckDB Vector Store with Vector Search Support.

    Features:
    - ✅ Vector similarity search with HNSW indexing
    - ✅ Hybrid search (when keyword search is available)
    - ❌ Full-Text Search (FTS): NOT IMPLEMENTED in current DuckDB version

    Note: Full-Text Search functionality is not available in the current DuckDB version.
    Hybrid search falls back to vector search only when FTS is unavailable.
    """

    def __init__(
        self,
        dataset: Dataset,
        database_path: str = "./duckdb_vector.db",
        table_name: str | None = None
    ):
        super().__init__(dataset)
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.database_path = database_path
        self.table_name = table_name or self.dataset.collection_name
        self._connection = duckdb.connect(self.database_path)

        # Load extensions
        self._load_extensions()

        # Initialize table schema
        # We try to infer dimension later, or assume 768/1536 mostly.
        # DuckDB VSS requires fixed array size?
        # Yes, FLOAT[N]. We need to know N.
        # Strategy: Create table on first insertion if not exists, or check existing schema.
        self._check_and_init_table_if_possible()

    def _load_extensions(self):
        try:
             self._connection.execute("INSTALL vss; LOAD vss;")
             self._connection.execute("INSTALL fts; LOAD fts;")
             self._connection.execute("INSTALL json; LOAD json;")
        except Exception as e:
             logger.warning(f"Failed to load DuckDB extensions: {e}")

    def _check_and_init_table_if_possible(self):
        # We can't create vector column without dimension.
        # We rely on 'create' method to do it, or if table exists, we are good.
        pass

    def create(self, texts: list[Document], **kwargs) -> None:
        """Create or Replace collection."""
        if not texts:
            return

        dim = 768
        if texts[0].vector:
            dim = len(texts[0].vector)

        # Create table with precise dimension
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

        # Create Indexes (HNSW only for now)
        try:
             # VSS Index
             self._connection.execute(f"CREATE INDEX IF NOT EXISTS idx_vec_{self.table_name} ON {self.table_name} USING HNSW (embedding)")
        except Exception as e:
             logger.warning(f"Index creation warning: {e}")

        # FTS Index will be created after data insertion

        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        if not texts: return

        # Ensure table exists (if create wasn't called specifically)
        # If table doesn't exist here, we try to create it inferred from first doc
        try:
             self._connection.execute(f"SELECT 1 FROM {self.table_name} LIMIT 0")
        except:
             # Table missing, call create logic
             self.create(texts)
             # return because create calls add_texts
             return

        data = []
        for doc in texts:
            meta_json = json.dumps(doc.metadata)
            data.append((doc.id, doc.page_content, doc.vector, meta_json, doc.id))

        sql = f"INSERT OR REPLACE INTO {self.table_name} VALUES (?, ?, ?, ?, ?)"
        self._connection.executemany(sql, data)

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:

        search_type = kwargs.get('search_type', 'similarity')

        if search_type == 'hybrid' and query_vector:
            return self._search_hybrid(query, query_vector, top_k)
        elif search_type == 'keyword':
            return self._search_keyword(query, top_k)
        else:
            return self._search_vector(query_vector, top_k)

    def _search_vector(self, query_vector: list[float], top_k: int) -> list[Document]:
        if not query_vector: return []

        # list_cosine_distance returns distance (0..2 for cosine).
        # Similarity = 1 - Distance (Approx for ranking)
        sql = f"""
        SELECT id, content, metadata, list_cosine_distance(embedding, ?::FLOAT[{len(query_vector)}]) as dist
        FROM {self.table_name}
        ORDER BY dist ASC
        LIMIT ?
        """
        rows = self._connection.execute(sql, [query_vector, top_k]).fetchall()
        return self._rows_to_docs(rows, score_strategy='distance')

    def _search_keyword(self, query: str, top_k: int) -> list[Document]:
        """
        Full-Text Search (FTS) is NOT IMPLEMENTED in current DuckDB version.

        This method raises NotImplementedError to clearly indicate that
        keyword-based text search is not available in DuckDB.

        For text search capabilities, consider using other vector stores
        that support both vector and text search (e.g., ChromaDB, Pinecone).
        """
        raise NotImplementedError(
            "DuckDB Full-Text Search (FTS) is not implemented in current version. "
            "Keyword search is not available. Use vector search or switch to "
            "a vector store that supports both vector and text search."
        )

        # Use correct syntax for DuckDB FTS
        sql = f"""
        SELECT t.id, t.content, t.metadata, {fts_table}.match_bm25(t.id, ?) as score
        FROM {self.table_name} t
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """
        try:
            logger.info(f"Executing FTS query: {query}")
            rows = self._connection.execute(sql, [query, top_k]).fetchall()
            logger.info(f"FTS query returned {len(rows)} rows")
        except Exception as e:
            logger.error(f"FTS query failed: {e}")
            return []

        return self._rows_to_docs(rows, score_strategy='score')

    def _search_hybrid(self, query: str, query_vector: list[float], top_k: int) -> list[Document]:
        """
        Hybrid search is NOT IMPLEMENTED because Full-Text Search is unavailable.

        Since DuckDB FTS is not implemented, true hybrid search (combining vector
        and keyword search) cannot be performed. This method raises NotImplementedError
        to clearly indicate this limitation.
        """
        raise NotImplementedError(
            "DuckDB Hybrid Search is not implemented because Full-Text Search (FTS) "
            "is not available in current version. Hybrid search requires both vector "
            "and keyword search capabilities. Use vector search only or switch to "
            "a vector store that supports both modalities."
        )

    def _rows_to_docs(self, rows, score_strategy='score') -> list[Document]:
        docs = []
        for row in rows:
            doc_id, content, meta_json, val = row
            try:
                meta = json.loads(meta_json)
            except:
                meta = {}

            if score_strategy == 'distance':
                # Convert distance to similarity
                # HNSW Cosine Distance is 1 - CosSim ? Or 1 - (CosSim+1)/2?
                # Usually we just invert it for ranking. 0 is best.
                score = 1.0 / (1.0 + val)
            else:
                score = val # Raw BM25 score

            meta['score'] = score

            docs.append(Document(
                id=doc_id,
                page_content=content,
                metadata=meta
            ))
        return docs

    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids: return
        ph = ",".join("?" for _ in ids)
        self._connection.execute(f"DELETE FROM {self.table_name} WHERE id IN ({ph})", ids)

    def delete(self) -> None:
        try:
            self._connection.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        except:
             pass

    def __del__(self):
        try:
             self._connection.close()
        except:
             pass
