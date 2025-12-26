"""Vector store capability declarations."""

from dataclasses import dataclass
from enum import Enum


class SearchMode(str, Enum):
    """Search mode for vector store queries.

    Attributes:
        VECTOR: Pure vector similarity search
        FULLTEXT: Pure full-text keyword search
        HYBRID: Combined vector + text search (either native or via RRF)
    """
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class VectorStoreCapabilities:
    """Declares what search capabilities a vector store supports.

    This allows the framework to determine which search modes are available
    and whether to use native implementations or fallback to combining
    multiple stores with RRF (Reciprocal Rank Fusion).

    Attributes:
        supports_vector: Supports vector similarity search (cosine, L2, etc)
        supports_fulltext: Supports full-text keyword search (BM25, etc)
        supports_hybrid: Supports native hybrid search combining both modes

    Examples:
        InMemoryVectorStore: VectorStoreCapabilities(vector=True, fulltext=False, hybrid=False)
        SeekDB: VectorStoreCapabilities(vector=True, fulltext=True, hybrid=True)
        Chroma: VectorStoreCapabilities(vector=True, fulltext=False, hybrid=False)
    """
    supports_vector: bool = True
    supports_fulltext: bool = False
    supports_hybrid: bool = False

    def validate_mode(self, mode: SearchMode) -> None:
        """Validate that this store supports the requested search mode.

        Args:
            mode: Requested search mode

        Raises:
            ValueError: If the mode is not supported
        """
        if mode == SearchMode.VECTOR and not self.supports_vector:
            raise ValueError("Vector search not supported by this store")
        if mode == SearchMode.FULLTEXT and not self.supports_fulltext:
            raise ValueError("Full-text search not supported by this store")
        if mode == SearchMode.HYBRID and not self.supports_hybrid:
            raise ValueError("Hybrid search not supported by this store")
