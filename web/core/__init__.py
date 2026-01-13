"""
Core module for the LangRAG Web Application.

This module provides the essential components for integrating LangRAG
into a web application:

- RAGKernel: Central coordinator for all RAG operations
- Embedders: WebOpenAIEmbedder, SeekDBEmbedder
- Services: DocumentProcessor, RetrievalService
- Managers: WebVectorStoreManager, WebLLMAdapter

Example:
    from web.core import RAGKernel

    kernel = RAGKernel()
    kernel.set_embedder("openai", model="...", base_url="...", api_key="...")
    kernel.set_llm(base_url="...", api_key="...", model="gpt-4")
"""

from .database import get_session, init_db
from .services.document_processor import DocumentProcessor
from .embedders import SeekDBEmbedder, WebOpenAIEmbedder
from .rag_kernel import RAGKernel
from .services.retrieval_service import RetrievalService
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
    # Services
    "DocumentProcessor",
    "RetrievalService",
    # Managers
    "WebVectorStoreManager",
]
