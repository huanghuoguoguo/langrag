from abc import ABC, abstractmethod
from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

class BaseIndexProcessor(ABC):
    """Abstract base class for Index Processors."""

    @abstractmethod
    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        """
        Process documents and index them into the datasource.
        
        Args:
            dataset: The target dataset.
            documents: List of raw documents to process.
        """
        pass
