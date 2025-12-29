from abc import ABC, abstractmethod
from typing import List
from langrag.entities.dataset import Dataset

class BaseRouter(ABC):
    """Abstract base class for Routing."""
    
    @abstractmethod
    def route(self, query: str, datasets: List[Dataset]) -> List[Dataset]:
        """
        Route the query to appropriate datasets.
        Returns a subset of datasets to query.
        """
        pass
