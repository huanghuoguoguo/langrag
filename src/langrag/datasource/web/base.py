from abc import ABC, abstractmethod

from pydantic import BaseModel


class WebSearchResult(BaseModel):
    """Result from a web search."""
    title: str
    link: str
    snippet: str
    source: str = "web"
    score: float = 0.0 # Relevance score if provided by engine

class BaseWebSearchProvider(ABC):
    """Abstract base class for web search providers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> list[WebSearchResult]:
        """
        Perform a web search.
        
        Args:
           query: The search string.
           top_k: Number of results to return.
           
        Returns:
           List of WebSearchResult objects.
        """
        pass
