from abc import ABC, abstractmethod

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class BaseIndexProcessor(ABC):
    """Abstract base class for Index Processors.

    This class defines the interface that all index processors must implement.
    Index processors are responsible for transforming raw documents into indexed
    representations suitable for retrieval.

    Supports both sync and async implementations:
    - Override `process()` for sync implementations (traditional usage)
    - Override `process_async()` for async implementations (e.g., remote APIs, IPC)

    The default `process_async()` wraps `process()` for backward compatibility.
    """

    @abstractmethod
    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        """
        Process documents and index them into the datasource (sync version).

        Args:
            dataset: The target dataset.
            documents: List of raw documents to process.
            **kwargs: Additional processing options.
        """
        pass

    async def process_async(
        self,
        dataset: Dataset,
        documents: list[Document],
        **kwargs
    ) -> None:
        """
        Process documents and index them into the datasource (async version).

        Override this method for async implementations (e.g., remote API calls,
        plugin IPC). Default implementation wraps the sync `process()` method.

        Args:
            dataset: The target dataset.
            documents: List of raw documents to process.
            **kwargs: Additional processing options.
        """
        import asyncio
        await asyncio.to_thread(self.process, dataset, documents, **kwargs)

