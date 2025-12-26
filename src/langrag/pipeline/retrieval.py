"""Retrieval pipeline for query processing."""

from loguru import logger

from ..embedder import BaseEmbedder
from ..vector_store import BaseVectorStore
from ..reranker import BaseReranker
from ..core.query import Query
from ..core.search_result import SearchResult


class RetrievalPipeline:
    """Pipeline for retrieving relevant chunks for a query.

    This pipeline orchestrates the complete retrieval workflow:
    1. Embed the query text
    2. Search vector store for similar chunks
    3. Optionally rerank results

    Attributes:
        embedder: Embedding generator
        vector_store: Vector storage backend
        reranker: Optional reranker for improving results
        top_k: Number of results to retrieve from vector search
        rerank_top_k: Number of results to return after reranking
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: BaseReranker | None = None,
        top_k: int = 5,
        rerank_top_k: int | None = None,
    ):
        """Initialize the retrieval pipeline.

        Args:
            embedder: Embedder for query vectorization
            vector_store: Store for similarity search
            reranker: Optional reranker (None = no reranking)
            top_k: Number of results from vector search
            rerank_top_k: Number of results after reranking (None = all)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query_text: str) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query_text: The query string

        Returns:
            List of search results, sorted by relevance
        """
        logger.info(f"Retrieving for query: {query_text[:50]}...")

        # 1. Embed query
        query_embedding = self.embedder.embed([query_text])[0]
        query = Query(text=query_text, vector=query_embedding)

        # 2. Vector search
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=self.top_k
        )
        logger.debug(f"Vector search returned {len(results)} results")

        # 3. Optional reranking
        if self.reranker is not None:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=self.rerank_top_k
            )
            logger.debug(f"Reranking returned {len(results)} results")

        return results
