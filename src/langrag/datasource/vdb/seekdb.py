import asyncio
import logging
from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.vdb.base import BaseVector

logger = logging.getLogger(__name__)

try:
    import pyseekdb
    SEEKDB_AVAILABLE = True
except ImportError:
    SEEKDB_AVAILABLE = False

class SeekDBVector(BaseVector):
    """
    SeekDB Vector Store Implementation.
    Supports Vector, Full-Text, and Hybrid search natively.
    """

    def __init__(
        self, 
        dataset: Dataset, 
        mode: str = "embedded",
        db_path: str = "./seekdb_data",
        host: str | None = None,
        port: int | None = None
    ):
        super().__init__(dataset)
        if not SEEKDB_AVAILABLE:
            raise ImportError("pyseekdb is required. Install with: pip install pyseekdb")

        self.mode = mode
        
        # Initialize Client
        if mode == "embedded":
             # Use current directory or config for embedded DB path
             # Need to ensure dataset.collection_name is used as DB name or Table name
             # Following legacy logic: db_name = collection_name
             self.db_name = self.collection_name 
             self._client = pyseekdb.Client(path=db_path, database=self.db_name)
             
             # Attempt to create DB if not exists (AdminClient needed)
             try:
                 admin = pyseekdb.AdminClient(path=db_path)
                 existing = [d.name for d in admin.list_databases()]
                 if self.db_name not in existing:
                     admin.create_database(self.db_name)
             except Exception as e:
                 logger.warning(f"Failed to check/create SeekDB database: {e}")

        else:
             if not host or not port:
                 raise ValueError("Host and port required for server mode")
             self._client = pyseekdb.Client(
                 host=host, 
                 port=port, 
                 database=self.collection_name, 
                 user="root", 
                 password=""
             )
        
    def create(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add texts."""
        # SeekDB creates collection via client.create_collection
        # We handle this lazily or explicitly.
        # Let's check existence first.
        if not self._client.has_collection(self.collection_name):
             from pyseekdb import HNSWConfiguration
             # Dimension needs to be known. In Dify it comes from EmbeddingModel. 
             # Here we might need to infer from texts or config.
             # Fallback to 768 or get from first text.
             dim = len(texts[0].vector) if texts and texts[0].vector else 768
             
             config = HNSWConfiguration(dimension=dim, distance="cosine")
             self._client.create_collection(
                 name=self.collection_name, 
                 configuration=config,
                 embedding_function=None # We handle embeddings
             )
        
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        if not texts: 
            return
            
        coll = self._client.get_collection(self.collection_name, embedding_function=None)
        
        ids = [doc.id for doc in texts]
        embeddings = [doc.vector for doc in texts]
        
        metadatas = []
        for doc in texts:
            # Copy metadata and inject content for retrieval (SeekDB might store content in separate column or metadata)
            # Legacy code stored content in metadata.
            m = doc.metadata.copy()
            m['content'] = doc.page_content # Store content so we can retrieve it
            metadatas.append(m)

        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(
        self, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> list[Document]:
        
        coll = self._client.get_collection(self.collection_name, embedding_function=None)
        search_type = kwargs.get('search_type', 'similarity')

        documents = []

        if search_type == 'hybrid' and query_vector:
            # Native Hybrid
            # SeekDB hybrid_search(query={'where_document':...}, knn={'query_embeddings':...})
            res = coll.hybrid_search(
                query={"where_document": {"$contains": query}, "n_results": top_k},
                knn={"query_embeddings": [query_vector], "n_results": top_k},
                n_results=top_k,
                include=["metadatas", "distances"]
            )
        elif search_type == 'keyword':
            # Full Text
             res = coll.get(
                where_document={"$contains": query},
                limit=top_k,
                include=["metadatas"]
            )
             # Fill fake distances for standard format
             res['distances'] = [[0.0] * len(res['ids'])] if 'ids' in res else []
        else:
            # Vector Search
            if not query_vector:
                return []
            res = coll.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["metadatas", "distances"]
            )

        # Parse Results
        if not res or not res.get('ids'):
            return []
            
        # Handle nested lists
        ids = res['ids'][0] if isinstance(res['ids'][0], list) else res['ids']
        metas = res['metadatas'][0] if isinstance(res['metadatas'][0], list) else res['metadatas']
        # Distances might be empty for keyword search
        dists = res.get('distances', [[]])
        dists = dists[0] if dists and isinstance(dists[0], list) else (dists or [0]*len(ids))

        for i, doc_id in enumerate(ids):
            meta = metas[i]
            score = 1.0 / (1.0 + float(dists[i])) if dists[i] is not None else 0.0
            
            # Reconstruct Document
            content = meta.pop('content', '') # Extract content
            meta['score'] = score
            
            doc = Document(
                id=str(doc_id),
                page_content=content,
                metadata=meta
            )
            documents.append(doc)
            
        return documents

    def delete_by_ids(self, ids: list[str]) -> None:
        coll = self._client.get_collection(self.collection_name)
        for i in ids:
            coll.delete(where={'id': i})

    def delete(self) -> None:
        self._client.delete_collection(self.collection_name)
