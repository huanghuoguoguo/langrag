"""Context compressor for RAG systems.

Context compression module for compressing context after retrieval and reranking
to reduce the number of tokens passed to the LLM.
"""

from .base import BaseCompressor
from .factory import CompressorFactory

__all__ = ["BaseCompressor", "CompressorFactory"]
