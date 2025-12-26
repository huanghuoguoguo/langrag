"""Context compressor for RAG systems.

上下文压缩模块，用于在检索和重排序后压缩上下文，减少传给 LLM 的 token 数量。
"""

from .base import BaseCompressor
from .factory import CompressorFactory

__all__ = ["BaseCompressor", "CompressorFactory"]
