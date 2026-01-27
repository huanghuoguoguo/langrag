from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset


class BaseRouter(ABC):
    """Abstract base class for Routing.

    This interface supports both sync and async implementations:
    - Override `route()` for sync implementations
    - Override `route_async()` for async implementations (LLM-based routers)

    The default `route_async()` wraps `route()` for backward compatibility.
    """

    @abstractmethod
    def route(self, query: str, datasets: list[Dataset]) -> list[Dataset]:
        """
        Route the query to appropriate datasets (sync version).

        Args:
            query: User query string
            datasets: List of available datasets

        Returns:
            Subset of datasets to query.
        """
        pass

    async def route_async(self, query: str, datasets: list[Dataset]) -> list[Dataset]:
        """
        Route the query to appropriate datasets (async version).

        Override this method for async implementations (e.g., LLM-based routers).
        Default implementation wraps the sync `route()` method.

        Args:
            query: User query string
            datasets: List of available datasets

        Returns:
            Subset of datasets to query.
        """
        import asyncio
        return await asyncio.to_thread(self.route, query, datasets)
