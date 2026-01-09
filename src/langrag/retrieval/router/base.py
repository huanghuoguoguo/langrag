from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset


class BaseRouter(ABC):
    """Abstract base class for Routing."""

    @abstractmethod
    def route(self, query: str, datasets: list[Dataset]) -> list[Dataset]:
        """
        Route the query to appropriate datasets.
        Returns a subset of datasets to query.
        """
        pass
