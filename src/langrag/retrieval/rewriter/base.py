from abc import ABC, abstractmethod


class BaseRewriter(ABC):
    """Abstract base class for Query Rewriting.

    This interface supports both sync and async implementations:
    - Override `rewrite()` for sync implementations
    - Override `rewrite_async()` for async implementations (LLM-based rewriters)

    The default `rewrite_async()` wraps `rewrite()` for backward compatibility.
    """

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """
        Rewrite the query for optimization (sync version).

        Args:
            query: Original query string

        Returns:
            Rewritten query string
        """
        pass

    async def rewrite_async(self, query: str) -> str:
        """
        Rewrite the query for optimization (async version).

        Override this method for async implementations (e.g., LLM-based rewriters).
        Default implementation wraps the sync `rewrite()` method.

        Args:
            query: Original query string

        Returns:
            Rewritten query string
        """
        import asyncio
        return await asyncio.to_thread(self.rewrite, query)
