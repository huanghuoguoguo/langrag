"""Chunker module for text splitting.

This module provides text chunking functionality with multiple
implementations and a factory for creating chunkers.
"""

from .base import BaseChunker
from .providers.fixed_size import FixedSizeChunker
from .factory import ChunkerFactory

__all__ = ["BaseChunker", "FixedSizeChunker", "ChunkerFactory"]
