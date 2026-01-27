"""Base reranker interface."""

from abc import ABC, abstractmethod

from langrag.entities.search_result import SearchResult


class BaseReranker(ABC):
    """Abstract base class for result reranking.

    Rerankers improve initial retrieval results by applying
    more sophisticated relevance scoring.

    This interface supports both sync and async implementations:
    - Override `rerank()` for sync implementations (local models)
    - Override `rerank_async()` for async implementations (remote APIs)

    The default `rerank_async()` wraps `rerank()` for backward compatibility.
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Rerank search results (sync version).

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked results, potentially truncated to top_k
        """
        pass

    async def rerank_async(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Rerank search results (async version).

        Override this method for async implementations (e.g., remote API calls).
        Default implementation wraps the sync `rerank()` method.

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked results, potentially truncated to top_k
        """
        import asyncio
        return await asyncio.to_thread(self.rerank, query, results, top_k)
