"""检索模块

提供统一的检索接口和多种检索策略实现。
"""

from .base import BaseRetrievalProvider
from .factory import ProviderFactory
from .providers.fulltext import FullTextSearchProvider
from .providers.hybrid import HybridSearchProvider
from .providers.vector import VectorSearchProvider
from .retriever import Retriever

__all__ = [
    "BaseRetrievalProvider",
    "VectorSearchProvider",
    "FullTextSearchProvider",
    "HybridSearchProvider",
    "ProviderFactory",
    "Retriever",
]
