"""Document entity representing a source document."""

from typing import Any
from pydantic import BaseModel, Field
from uuid import uuid4


class Document(BaseModel):
    """Represents a source document in the RAG system.

    Attributes:
        id: Unique identifier (auto-generated UUID if not provided)
        content: The text content of the document
        metadata: Flexible dictionary for document-level metadata
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "str_strip_whitespace": True,
        "frozen": False,
    }
