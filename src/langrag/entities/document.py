"""Document entity representing a source document or a chunk."""

from enum import StrEnum

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    ORIGINAL = "original"  # The full source document
    CHUNK = "chunk"        # A text chunk for vector search
    PARENT = "parent"      # A parent chunk in Parent-Child indexing


class Document(BaseModel):
    """
    Represents a source document OR a chunk in the RAG system.
    
    This unified model allows for easier handling across different layers.
    """
    # Content
    page_content: str = Field(..., min_length=1)

    # Vector Representation (Optional, only present after embedding)
    vector: list[float] | None = Field(default=None)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Core IDs (Extracted from metadata for easier access, but sync is manual)
    id: str = Field(default_factory=lambda: str(uuid4())) # The chunk ID (or doc ID)

    # Type identifier
    type: DocumentType = Field(default=DocumentType.ORIGINAL)

    # Dify Compatibility Fields (Often stored in metadata)
    # dataset_id: str
    # document_id: str (original file ID)
    # doc_id: str (segment/chunk ID, usually same as self.id)

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True
    }
