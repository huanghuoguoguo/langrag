from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """
    Represents a knowledge base (Dataset).

    In Dify/LangRAG architecture, a Dataset is the logical container for documents.
    It dictates the indexing strategy and storage location.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str | None = None
    
    # Configuration
    indexing_technique: str = Field(default="high_quality")  # "high_quality" | "economy"
    collection_name: str  # The physical table/index name in the vector store
    
    # Metadata
    tenant_id: str | None = None
    created_at: int | None = None
    
    model_config = {
        "frozen": False
    }

class RetrievalContext(BaseModel):
    """
    Context returned after retrieval used for generation.
    """
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]
