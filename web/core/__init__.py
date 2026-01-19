"""
Core module for the LangRAG Web Application.

This module provides the essential components for integrating LangRAG
into a web application:

- RAGKernel: Central coordinator for all RAG operations
- Embedders: WebOpenAIEmbedder, SeekDBEmbedder
- Managers: WebVectorStoreManager

Example:
    from web.core import RAGKernel

    kernel = RAGKernel()
    kernel.set_embedder("openai", model="...", base_url="...", api_key="...")
    kernel.add_llm("gpt-4", {...}, set_as_default=True)
"""

from .database import get_session, init_db
from .embedders import SeekDBEmbedder, WebOpenAIEmbedder
from .rag_kernel import RAGKernel
from .vdb_manager import WebVectorStoreManager

__all__ = [
    # Database
    "get_session",
    "init_db",
    # Main Kernel
    "RAGKernel",
    # Embedders
    "WebOpenAIEmbedder",
    "SeekDBEmbedder",
    # Managers
    "WebVectorStoreManager",
]
