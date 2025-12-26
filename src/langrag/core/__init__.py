"""Core data entities for LangRAG."""

from .document import Document
from .chunk import Chunk
from .query import Query
from .search_result import SearchResult

__all__ = ["Document", "Chunk", "Query", "SearchResult"]
