"""API routers."""

from .config import router as config_router
from .document import router as document_router
from .kb import router as kb_router
from .search import router as search_router

__all__ = ["kb_router", "document_router", "search_router", "config_router"]
