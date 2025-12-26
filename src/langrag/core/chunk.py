"""Chunk entity representing a segment of a document."""

from typing import Any
from pydantic import BaseModel, Field
from uuid import uuid4


class Chunk(BaseModel):
    """Represents a chunk of text from a document.

    Attributes:
        id: Unique identifier (auto-generated UUID if not provided)
        content: The text content of this chunk
        embedding: Optional vector embedding (populated by pipeline)
        metadata: Flexible dictionary for chunk-level metadata
        source_doc_id: Reference to the source document ID
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., min_length=1)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str

    model_config = {
        "str_strip_whitespace": True,
        "frozen": False,
    }
