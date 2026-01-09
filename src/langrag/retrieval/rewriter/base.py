from abc import ABC, abstractmethod


class BaseRewriter(ABC):
    """Abstract base class for Query Rewriting."""

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """
        Rewrite the query optimization.
        """
        pass
