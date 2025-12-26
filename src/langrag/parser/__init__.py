"""Parser module for document parsing.

This module provides document parsing functionality with multiple
implementations and a factory for creating parsers.
"""

from .base import BaseParser
from .providers.simple_text import SimpleTextParser
from .factory import ParserFactory

__all__ = ["BaseParser", "SimpleTextParser", "ParserFactory"]
