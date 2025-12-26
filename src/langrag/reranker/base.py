"""Base reranker interface."""

from abc import ABC, abstractmethod

from ..core.query import Query
from ..core.search_result import SearchResult


class BaseReranker(ABC):
    """Abstract base class for result reranking.

    Rerankers improve initial retrieval results by applying
    more sophisticated relevance scoring.
    """

    @abstractmethod
    def rerank(
        self, query: Query, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Rerank search results.

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked results, potentially truncated to top_k
        """
        pass
