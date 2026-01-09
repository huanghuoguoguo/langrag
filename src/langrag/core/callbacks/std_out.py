from typing import Any
from uuid import UUID

from loguru import logger

from .base import BaseCallbackHandler


class StdOutCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to stdout using loguru."""

    def on_retrieve_start(
        self, query: str, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        logger.info(f"[Callback] Retrieve Start: query='{query}' (run_id={run_id})")

    def on_retrieve_end(
        self, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        logger.info(f"[Callback] Retrieve End: Found {len(documents)} documents (run_id={run_id})")

    def on_rerank_start(
        self, query: str, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        logger.info(f"[Callback] Rerank Start: {len(documents)} documents (run_id={run_id})")

    def on_rerank_end(
        self, documents: list[Any], run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        logger.info(f"[Callback] Rerank End: {len(documents)} documents retained (run_id={run_id})")

    def on_error(
        self, error: Exception, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        logger.error(f"[Callback] Error: {error} (run_id={run_id})")
