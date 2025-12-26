"""Base LLM interface."""

from abc import ABC, abstractmethod

from ..core.search_result import SearchResult


class BaseLLM(ABC):
    """Abstract base class for LLM-based generation.

    LLMs generate responses using retrieved context.
    """

    @abstractmethod
    def generate(self, query: str, context: list[SearchResult], **kwargs) -> str:
        """Generate response using query and context.

        Args:
            query: User query string
            context: Retrieved context chunks
            **kwargs: Model-specific generation parameters

        Returns:
            Generated response text
        """
        pass
