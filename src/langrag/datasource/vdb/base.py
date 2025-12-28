from abc import ABC, abstractmethod
from typing import Any

from langrag.entities.document import Document
from langrag.entities.dataset import Dataset

class BaseVector(ABC):
    """Abstract base class for Vector Database implementations."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.collection_name = dataset.collection_name

    @abstractmethod
    def create(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add texts (blocks until done)."""
        pass

    @abstractmethod
    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Add texts to existing collection."""
        pass

    @abstractmethod
    def search(
        self, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> list[Document]:
        """
        Search for documents.
        
        Args:
            query: The raw text query.
            query_vector: The embedded query vector (optional, depends on impl).
            top_k: Number of results.
            kwargs: 
                - search_type: "similarity", "mmr", "hybrid"
                - filter: Metadata filters
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        pass
    
    @abstractmethod
    def delete(self) -> None:
        """Delete the entire collection."""
        pass
