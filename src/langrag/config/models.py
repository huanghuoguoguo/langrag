"""Configuration models for RAG components.

This module defines Pydantic models for component configuration.
All components are configured via a type string and optional parameters.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StorageRole(str, Enum):
    """Storage role definition for multi-datasource scenarios.

    Attributes:
        PRIMARY: Primary storage, stores complete data (chunks + embeddings + metadata)
        VECTOR_ONLY: Vector-only storage, stores only embeddings and basic metadata
        FULLTEXT_ONLY: Fulltext-only storage, stores only text content for keyword retrieval
        BACKUP: Backup storage, complete redundant backup
    """

    PRIMARY = "primary"
    VECTOR_ONLY = "vector_only"
    FULLTEXT_ONLY = "fulltext_only"
    BACKUP = "backup"


class ComponentConfig(BaseModel):
    """Configuration for a single component.

    Attributes:
        type: Component type identifier (e.g., "simple_text", "fixed_size")
        params: Component-specific parameters as a dictionary
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(ComponentConfig):
    """Extended vector store configuration with storage role support.

    Attributes:
        type: Storage type (e.g., "seekdb", "chroma", "duckdb")
        params: Storage parameters
        role: Storage role (for multi-datasource scenarios)
        enabled: Whether this storage is enabled (for dynamic switching)
    """

    role: StorageRole = StorageRole.PRIMARY
    enabled: bool = True


class RetrievalConfig(BaseModel):
    """Retrieval configuration.

    Attributes:
        mode: Retrieval mode ("single" | "multi_store" | "auto")
        fusion_strategy: Multi-source fusion strategy ("rrf" | "weighted_rrf" | "linear")
        fusion_weights: Fusion weights (for weighted_rrf)
        top_k_per_store: Number of results to retrieve per store
        final_top_k: Final number of results to return
    """

    mode: str = "auto"  # "single" | "multi_store" | "auto"
    fusion_strategy: str = "rrf"  # "rrf" | "weighted_rrf" | "linear"
    fusion_weights: list[float] | None = None
    top_k_per_store: int = 10
    final_top_k: int = 5


class RAGConfig(BaseModel):
    """Main RAG system configuration.

    Attributes:
        parser: Parser component configuration
        chunker: Chunker component configuration
        embedder: Embedder component configuration
        vector_store: Single vector store configuration (backward compatible)
        vector_stores: Multiple vector store configurations (new)
        reranker: Optional reranker component configuration
        llm: Optional LLM component configuration
        retrieval: Retrieval configuration
        retrieval_top_k: Number of results to retrieve from vector search (deprecated)
        rerank_top_k: Number of results to return after reranking (deprecated)
    """

    parser: ComponentConfig
    chunker: ComponentConfig
    embedder: ComponentConfig

    # Vector store configuration: supports single or multiple
    vector_store: VectorStoreConfig | None = None  # Single store (backward compatible)
    vector_stores: list[VectorStoreConfig] | None = None  # Multiple stores (new)

    reranker: ComponentConfig | None = None
    compressor: ComponentConfig | None = None  # Context compressor configuration
    llm: ComponentConfig | None = None

    # Retrieval configuration
    retrieval: RetrievalConfig | None = None
    compression_ratio: float = Field(default=0.5, ge=0.1, le=1.0)  # Compression ratio (0.1-1.0)

    # Pipeline settings (deprecated, use retrieval config instead)
    retrieval_top_k: int = Field(default=5, ge=1)
    rerank_top_k: int | None = Field(default=None, ge=1)

    model_config = {
        "extra": "allow",  # Allow additional fields for extensibility
    }

    def get_vector_stores(self) -> list[VectorStoreConfig]:
        """Get all vector store configurations (handles both single and multiple cases).

        Returns:
            List of vector store configurations
        """
        if self.vector_stores:
            return [vs for vs in self.vector_stores if vs.enabled]
        elif self.vector_store:
            return [self.vector_store]
        return []

    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration (backward compatible with old config).

        Returns:
            Retrieval configuration object
        """
        if self.retrieval:
            return self.retrieval

        # Backward compatible: build from old config
        return RetrievalConfig(
            mode="auto",
            final_top_k=self.rerank_top_k or self.retrieval_top_k,
            top_k_per_store=self.retrieval_top_k,
        )
