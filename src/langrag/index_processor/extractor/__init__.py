"""Parser module for document parsing.

This module provides document parsing functionality with multiple
implementations and a factory for creating parsers.
"""

from .base import BaseParser
from .factory import ParserFactory
from .providers.docx import DocxParser
from .providers.html import HtmlParser
from .providers.markdown import MarkdownParser
from .providers.pdf import FileTooLargeError, PdfParser
from .providers.simple_text import SimpleTextParser

__all__ = [
    "BaseParser",
    "SimpleTextParser",
    "PdfParser",
    "DocxParser",
    "MarkdownParser",
    "HtmlParser",
    "ParserFactory",
    "FileTooLargeError",
]
