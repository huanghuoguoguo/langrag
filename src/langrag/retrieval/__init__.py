"""Retrieval module.

Provides the retrieval workflow and related components.

Components:
- RetrievalWorkflow: Main orchestrator for the retrieval pipeline
- WorkflowConfig: Configuration for the workflow
- RetrievalExecutor: Handles parallel/single dataset retrieval
- PostProcessor: Deduplication and filtering
- GraphRetriever: Knowledge graph-based retrieval for GraphRAG
"""

from .config import (
    WorkflowConfig,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRIEVAL_TIMEOUT,
    DEFAULT_ROUTER_TIMEOUT,
)
from .executor import RetrievalExecutor
from .post_processor import PostProcessor
from .workflow import RetrievalWorkflow
from .graph import GraphRetriever, GraphRetrieverConfig

__all__ = [
    "RetrievalWorkflow",
    "WorkflowConfig",
    "RetrievalExecutor",
    "PostProcessor",
    "GraphRetriever",
    "GraphRetrieverConfig",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_RETRIEVAL_TIMEOUT",
    "DEFAULT_ROUTER_TIMEOUT",
]
