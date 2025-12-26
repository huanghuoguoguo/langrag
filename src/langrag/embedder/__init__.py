"""Embedder module for vector generation.

This module provides embedding functionality with multiple
implementations and a factory for creating embedders.
"""

from .base import BaseEmbedder
from .factory import EmbedderFactory
from .providers.mock import MockEmbedder

__all__ = ["BaseEmbedder", "MockEmbedder", "EmbedderFactory"]
