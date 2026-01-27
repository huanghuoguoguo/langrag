from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class BaseVector(ABC):
    """Abstract base class for Vector Database implementations.

    This interface supports both sync and async implementations:
    - Override sync methods for local/embedded databases
    - Override async methods (`*_async`) for remote database calls

    The default async methods wrap sync methods for backward compatibility.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.collection_name = dataset.collection_name

    # ==================== Sync Methods ====================

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
                - search_type: "similarity", "keyword", "hybrid"
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

    # ==================== Async Methods ====================

    async def create_async(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add texts (async version).

        Override for async implementations. Default wraps sync method.
        """
        import asyncio
        return await asyncio.to_thread(self.create, texts, **kwargs)

    async def add_texts_async(self, texts: list[Document], **kwargs) -> None:
        """Add texts to existing collection (async version).

        Override for async implementations. Default wraps sync method.
        """
        import asyncio
        return await asyncio.to_thread(self.add_texts, texts, **kwargs)

    async def search_async(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Search for documents (async version).

        Override for async implementations. Default wraps sync method.

        Args:
            query: The raw text query.
            query_vector: The embedded query vector (optional).
            top_k: Number of results.
            kwargs:
                - search_type: "similarity", "keyword", "hybrid"
                - filter: Metadata filters
        """
        import asyncio
        return await asyncio.to_thread(self.search, query, query_vector, top_k, **kwargs)

    async def delete_by_ids_async(self, ids: list[str]) -> None:
        """Delete documents by their IDs (async version).

        Override for async implementations. Default wraps sync method.
        """
        import asyncio
        return await asyncio.to_thread(self.delete_by_ids, ids)

    async def delete_async(self) -> None:
        """Delete the entire collection (async version).

        Override for async implementations. Default wraps sync method.
        """
        import asyncio
        return await asyncio.to_thread(self.delete)
