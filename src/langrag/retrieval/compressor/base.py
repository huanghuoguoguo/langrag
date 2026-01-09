"""Base class for context compressors."""

from abc import ABC, abstractmethod

from langrag.entities.search_result import SearchResult


class BaseCompressor(ABC):
    """Abstract base class for context compressors.

    Base class for context compressors, used to compress retrieval results
    before passing to LLM to reduce token count.

    Typical use cases:
    - Extract key sentences
    - Filter redundant content
    - Summarize long text
    """

    @abstractmethod
    def compress(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """Compress the context content of retrieval results.

        Args:
            query: User query
            results: List of retrieval results
            target_ratio: Target compression ratio (0-1), e.g., 0.5 means compress to 50% length

        Returns:
            List of compressed retrieval results, each result's content may be compressed
        """
        pass
