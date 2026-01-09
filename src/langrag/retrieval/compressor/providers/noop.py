"""No-op compressor that returns results unchanged."""

from loguru import logger

from langrag.entities.search_result import SearchResult

from ..base import BaseCompressor


class NoOpCompressor(BaseCompressor):
    """Pass-through compressor that performs no compression.

    A placeholder implementation that performs no compression and returns
    the original results as-is.
    """

    def compress(
        self,
        query: str,  # noqa: ARG002
        results: list[SearchResult],
        target_ratio: float = 0.5,  # noqa: ARG002
    ) -> list[SearchResult]:
        """Return results unchanged.

        Args:
            query: User query (unused)
            results: List of retrieval results
            target_ratio: Target compression ratio (unused)

        Returns:
            Original results, uncompressed
        """
        logger.debug("NoOpCompressor: returning original results without compression")
        return results

