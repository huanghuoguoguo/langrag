"""No-op reranker that returns results unchanged."""

from loguru import logger

from ...core.query import Query
from ...core.search_result import SearchResult
from ..base import BaseReranker


class NoOpReranker(BaseReranker):
    """Pass-through reranker that performs no reranking.

    This reranker returns the original results unchanged,
    optionally truncating to top_k.
    """

    def rerank(
        self, query: Query, results: list[SearchResult], top_k: int | None = None  # noqa: ARG002
    ) -> list[SearchResult]:
        """Return results unchanged.

        Args:
            query: Original query (unused, but kept for interface compatibility)
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Original results, optionally truncated
        """
        logger.debug("NoOpReranker: returning original results")

        if top_k is not None:
            return results[:top_k]
        return results
