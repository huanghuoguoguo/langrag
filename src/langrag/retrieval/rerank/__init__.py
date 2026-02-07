"""Reranker module for result reordering.

This module provides reranking functionality with multiple
implementations and a factory for creating rerankers.
"""

from .base import BaseReranker
from .factory import RerankerFactory
from .providers.noop import NoOpReranker

# Optional providers (require httpx)
try:
    from .providers.cohere import CohereReranker
except ImportError:
    CohereReranker = None  # type: ignore

try:
    from .providers.llm_template import LLMTemplateReranker
except ImportError:
    LLMTemplateReranker = None  # type: ignore

try:
    from .providers.qwen import QwenReranker
except ImportError:
    QwenReranker = None  # type: ignore

__all__ = [
    "BaseReranker",
    "NoOpReranker",
    "RerankerFactory",
]

# Only export available providers
if CohereReranker is not None:
    __all__.append("CohereReranker")
if LLMTemplateReranker is not None:
    __all__.append("LLMTemplateReranker")
if QwenReranker is not None:
    __all__.append("QwenReranker")
