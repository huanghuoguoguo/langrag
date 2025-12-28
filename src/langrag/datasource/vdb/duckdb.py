import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.vdb.base import BaseVector
from langrag.utils.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class DuckDBVector(BaseVector):
    """
    All-in-One DuckDB Vector Store.
    Supports Vector, Full-Text, and Application-Side Hybrid search within a single DB.
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
        
        # Create Indexes (HNSW + FTS)
        try:
             # VSS Index
             self._connection.execute(f"CREATE INDEX IF NOT EXISTS idx_vec_{self.table_name} ON {self.table_name} USING HNSW (embedding)")
             # FTS Index
             self._connection.execute(f"PRAGMA create_fts_index('{self.table_name}', 'id', 'content')")
        except Exception as e:
             logger.warning(f"Index creation warning: {e}")

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
        if not query: return []
        fts_table = f"fts_main_{self.table_name}"
        sql = f"""
        SELECT t.id, t.content, t.metadata, {fts_table}.match_bm25(t.id, ?) as score
        FROM {self.table_name} t
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """
        rows = self._connection.execute(sql, [query, top_k]).fetchall()
        return self._rows_to_docs(rows, score_strategy='score')

    def _search_hybrid(self, query: str, query_vector: list[float], top_k: int) -> list[Document]:
        # Perform both searches
        vec_docs = self._search_vector(query_vector, top_k)
        kw_docs = self._search_keyword(query, top_k)
        
        # Prepare for RRF
        vec_list = [(d.id, d.metadata.get('score', 0)) for d in vec_docs]
        kw_list = [(d.id, d.metadata.get('score', 0)) for d in kw_docs]
        
        # Fusion
        fused = reciprocal_rank_fusion([vec_list, kw_list], k=60) # RRF default const
        
        # Map back to documents
        # We need a lookup dict
        all_docs_map = {d.id: d for d in vec_docs + kw_docs}
        
        final_results = []
        for doc_id, score in fused[:top_k]:
            if doc_id in all_docs_map:
                doc = all_docs_map[doc_id]
                # Update score to RRF score
                doc.metadata['score'] = score
                # Optionally mark it as hybrid
                doc.metadata['retrieval_method'] = 'hybrid'
                final_results.append(doc)
                
        return final_results

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
