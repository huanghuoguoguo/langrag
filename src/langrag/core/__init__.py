"""Core data entities for LangRAG."""

from .chunk import Chunk
from .document import Document
from .query import Query
from .search_result import SearchResult

__all__ = ["Document", "Chunk", "Query", "SearchResult"]
