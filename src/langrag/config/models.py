"""Configuration models for RAG system components."""

from typing import Any
from pydantic import BaseModel, Field


class ComponentConfig(BaseModel):
    """Configuration for a single component.

    Attributes:
        type: Component type identifier (e.g., "simple_text", "fixed_size")
        params: Dictionary of initialization parameters for the component
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Main RAG system configuration.

    Attributes:
        parser: Parser component configuration
        chunker: Chunker component configuration
        embedder: Embedder component configuration
        vector_store: Vector store component configuration
        reranker: Optional reranker component configuration
        llm: Optional LLM component configuration
        retrieval_top_k: Number of results to retrieve from vector search
        rerank_top_k: Number of results to return after reranking (optional)
    """

    parser: ComponentConfig
    chunker: ComponentConfig
    embedder: ComponentConfig
    vector_store: ComponentConfig
    reranker: ComponentConfig | None = None
    llm: ComponentConfig | None = None

    # Pipeline settings
    retrieval_top_k: int = Field(default=5, ge=1)
    rerank_top_k: int | None = Field(default=None, ge=1)

    model_config = {
        "extra": "allow",  # Allow additional fields for extensibility
    }
