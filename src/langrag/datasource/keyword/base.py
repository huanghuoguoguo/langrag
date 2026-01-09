from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class BaseKeyword(ABC):
    """Abstract base class for Keyword Search implementations."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def create(self, texts: list[Document], **kwargs) -> None:
        """Create index and add texts."""
        pass

    @abstractmethod
    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Add texts to existing index."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 4, **kwargs) -> list[Document]:
        """
        Perform keyword/full-text search.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        pass
