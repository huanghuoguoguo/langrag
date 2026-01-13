import contextlib
import logging
import os
from typing import Any

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

logger = logging.getLogger(__name__)

try:
    import pyseekdb
    # pyseekdb <= 1.0.0b6 doesn't have errors module, uses ValueError
    DatabaseNotFoundError = ValueError
    SEEKDB_AVAILABLE = True
except ImportError:
    SEEKDB_AVAILABLE = False
    DatabaseNotFoundError = Exception  # Fallback for type hints

# Default vector dimension (commonly used embedding models)
DEFAULT_VECTOR_DIMENSION = 768

class SeekDBVector(BaseVector):
    """
    SeekDB Vector Store Implementation.
    Supports Vector, Full-Text, and Hybrid search natively.
    """

    def __init__(
        self,
        dataset: Dataset,
        mode: str = "embedded",
        db_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None
    ):
        """
        Initialize SeekDB Vector Store.

        Args:
            dataset: Dataset configuration
            mode: "embedded" or "server"
            db_path: Path for embedded database (required for embedded mode)
            host: Server host (required for server mode)
            port: Server port (required for server mode)
            user: Username for server mode (defaults to SEEKDB_USER env var)
            password: Password for server mode (defaults to SEEKDB_PASSWORD env var)
        """
        super().__init__(dataset)
        if not SEEKDB_AVAILABLE:
            raise ImportError("pyseekdb is required. Install with: pip install pyseekdb")

        self.mode = mode

        # Validate db_path for embedded mode
        if mode == "embedded" and not db_path:
            raise ValueError("db_path is required for embedded mode")

        # Initialize Client
        self.db_path = db_path
        self.host = host
        self.port = port
        self.db_name = self.collection_name
        self._client_instance = None  # Lazy loaded
        self._closed = False

        # Load credentials from environment if not provided
        self._user = user or os.environ.get("SEEKDB_USER")
        self._password = password or os.environ.get("SEEKDB_PASSWORD", "")

        self._escape_table = str.maketrans({
            '\x00': '',
            '\\': '\\\\',
            '"': '\\"',
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t',
        })

    @property
    def _client(self):
        """Lazy load the client."""
        if self._client_instance:
            return self._client_instance

        if self._closed:
            raise RuntimeError(
                "Cannot access client: SeekDB connection has been closed. "
                "Create a new SeekDBVector instance to continue."
            )

        logger.info(f"Lazy initializing SeekDB client: {self.db_name}")

        if self.mode == "embedded":
            # Attempt to create DB if not exists (AdminClient needed)
            try:
                admin = pyseekdb.AdminClient(path=self.db_path)
                try:
                    admin.get_database(self.db_name)
                except DatabaseNotFoundError:
                    logger.info(f"Database '{self.db_name}' not found, creating...")
                    admin.create_database(self.db_name)
                del admin
            except DatabaseNotFoundError:
                raise  # Re-raise specific errors
            except Exception as e:
                logger.warning(f"Failed to check/create SeekDB database: {e}")

            self._client_instance = pyseekdb.Client(path=self.db_path, database=self.db_name)
        else:
            if not self.host or not self.port:
                raise ValueError("Host and port required for server mode")
            if not self._user:
                raise ValueError(
                    "Username required for server mode. "
                    "Set via 'user' parameter or SEEKDB_USER environment variable."
                )
            self._client_instance = pyseekdb.Client(
                host=self.host,
                port=self.port,
                database=self.collection_name,
                user=self._user,
                password=self._password
            )

        return self._client_instance

    def _clean_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """SeekDB metadata doesn't support \\ and ", insert will error 3104/3140"""
        return {
            k: v.translate(self._escape_table) if isinstance(v, str)
            else v if v is None or isinstance(v, (int, float, bool))
            else str(v)
            for k, v in meta.items()
            if v is not None
        }

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
             dim = len(texts[0].vector) if texts and texts[0].vector else DEFAULT_VECTOR_DIMENSION

             config = HNSWConfiguration(dimension=dim, distance="cosine")
             self._client.create_collection(
                 name=self.collection_name,
                 configuration=config,
                 embedding_function=None # We handle embeddings
             )

        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **_kwargs) -> None:
        if not texts:
            return

        # Check if collection exists, if not create it using the dimension from the first text
        if not self._client.has_collection(self.collection_name):
             from pyseekdb import HNSWConfiguration
             # Infer dimension from first text vector, default to 384 if not available
             dim = len(texts[0].vector) if texts and texts[0].vector else DEFAULT_VECTOR_DIMENSION

             logger.info(f"Creating SeekDB collection '{self.collection_name}' with dimension {dim}")
             config = HNSWConfiguration(dimension=dim, distance="cosine")
             self._client.create_collection(
                 name=self.collection_name,
                 configuration=config,
                 embedding_function=None # We handle embeddings externally
             )

        coll = self._client.get_collection(self.collection_name, embedding_function=None)

        ids = [doc.id for doc in texts]
        embeddings = [doc.vector for doc in texts]

        metadatas = []
        for doc in texts:
            # Copy metadata and inject content for retrieval
            m = doc.metadata.copy()
            m['content'] = doc.page_content # Store content so we can retrieve it
            metadatas.append(self._clean_metadata(m))

        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:

        if not self._client.has_collection(self.collection_name):
            logger.debug(f"Collection {self.collection_name} does not exist, returning empty results.")
            return []

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

    def close(self) -> None:
        """
        Close the underlying client connection.

        This method should be called when done using the vector store to ensure
        proper resource cleanup. Alternatively, use the context manager pattern.

        Note:
            After calling close(), the store instance cannot be used.
            Any subsequent operations will raise RuntimeError.
        """
        if self._closed:
            return

        try:
            if self._client_instance is not None:
                if hasattr(self._client_instance, 'close'):
                    self._client_instance.close()
                elif hasattr(self._client_instance, '__exit__'):
                    self._client_instance.__exit__(None, None, None)
                self._client_instance = None
            logger.debug(f"SeekDB connection closed for {self.db_name}")
        except Exception as e:
            logger.warning(f"Error closing SeekDB connection: {e}")
        finally:
            self._closed = True

    def __enter__(self) -> "SeekDBVector":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __del__(self):
        """Clean up connection on garbage collection."""
        if hasattr(self, '_closed') and not self._closed:
            with contextlib.suppress(Exception):
                self.close()
