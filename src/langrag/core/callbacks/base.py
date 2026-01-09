from abc import ABC
from typing import Any
from uuid import UUID


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to handle callbacks from LangRAG."""

    def on_retrieve_start(
        self, query: str, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when retrieval starts."""
        pass

    def on_retrieve_end(
        self, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when retrieval ends."""
        pass

    def on_rerank_start(
        self, query: str, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when reranking starts."""
        pass

    def on_rerank_end(
        self, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when reranking ends."""
        pass

    def on_llm_start(
        self, prompt: str, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when LLM starts."""
        pass

    def on_llm_end(
        self, response: str, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when LLM ends."""
        pass

    def on_error(
        self, error: Exception, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        """Run when error occurs."""
        pass
