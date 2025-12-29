from typing import Any, List, Optional
from uuid import UUID, uuid4
from loguru import logger
from .base import BaseCallbackHandler

class CallbackManager(BaseCallbackHandler):
    """Callback Manager that handles a list of callback handlers."""

    def __init__(self, handlers: List[BaseCallbackHandler]):
        self.handlers = handlers

    def add_handler(self, handler: BaseCallbackHandler):
        self.handlers.append(handler)

    def on_retrieve_start(
        self, query: str, run_id: UUID = None, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        run_id = run_id or uuid4()
        for handler in self.handlers:
            try:
                handler.on_retrieve_start(query, run_id, parent_run_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback handler {handler}: {e}")
        return run_id

    def on_retrieve_end(
        self, documents: List[Any], run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        for handler in self.handlers:
            try:
                handler.on_retrieve_end(documents, run_id, parent_run_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback handler {handler}: {e}")

    def on_rerank_start(
        self, query: str, documents: List[Any], run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        # Assuming run_id is passed
        for handler in self.handlers:
            try:
                handler.on_rerank_start(query, documents, run_id, parent_run_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback handler {handler}: {e}")

    def on_rerank_end(
        self, documents: List[Any], run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        for handler in self.handlers:
            try:
                handler.on_rerank_end(documents, run_id, parent_run_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback handler {handler}: {e}")

    def on_error(
        self, error: Exception, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        for handler in self.handlers:
            try:
                handler.on_error(error, run_id, parent_run_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback handler {handler}: {e}")
